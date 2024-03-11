# coding:utf-8 
import random
import time
import warnings
import sys
import argparse
import numpy
import os.path as osp
import os
import wandb
# import ot
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import torch.nn.functional as F
from imagelist import ImageList
from scipy.spatial.distance import cdist
import numpy as np
from torch.autograd import Variable

sys.path.append('../')
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss, ImageClassifier
from dalib.adaptation.mcc import MinimumClassConfusionLoss
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance
from common.utils.sam import SAM

sys.path.append('.')
import utils

FN = torch.from_numpy
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def find_max_indices(tensor):
    vector = []
    for row in tensor:
        max_value = torch.max(row).item()
        max_index = torch.argmax(row).item()
        vector.append(max_index)
    return vector


#################################

def calculate_accuracy(tensor1, tensor2):
    tensor1 = torch.tensor(tensor1).to(device)
    tensor2 = tensor2.to(device)
    if tensor1.shape != tensor2.shape:
        raise ValueError("The shapes of the input tensors must be the same.")

    num_correct = torch.sum(tensor1 == tensor2).item()
    accuracy = num_correct / tensor1.numel()
    return accuracy


def compute_lmmd(source_features, target_features):
    # 计算源域和目标域的局部最大均值差异（LMMD）

    # 计算源域和目标域的均值
    source_mean = torch.mean(source_features, dim=0)
    target_mean = torch.mean(target_features, dim=0)

    # 计算源域和目标域的中心化特征
    source_centered = source_features - source_mean
    target_centered = target_features - target_mean

    # 计算源域和目标域的局部最大均值差异（LMMD）
    lmmd = torch.mean(torch.norm(F.normalize(source_centered, dim=1) - F.normalize(target_centered, dim=1), dim=1))

    return lmmd


def entropy(t1, t2):
    # 计算每个张量的概率分布
    p1 = F.softmax(t1, dim=1)
    p2 = F.softmax(t2, dim=1)

    # 计算信息熵
    entropy1 = torch.sum(-p1 * torch.log2(p1), dim=1)
    entropy2 = torch.sum(-p2 * torch.log2(p2), dim=1)

    # 对信息熵结果进行求和或平均，得到标量输出
    entropy_sum = torch.sum(entropy1 + entropy2)
    # 或者使用平均值
    # entropy_mean = torch.mean(entropy1 + entropy2)

    return entropy_sum


def compute_kl_divergence(source_tensor, target_tensor):
    # 计算源域和目标域特征的均值和协方差矩阵
    source_mean = torch.mean(source_tensor, dim=0)
    source_cov = torch.matmul((source_tensor - source_mean).t(), (source_tensor - source_mean)) / (
            source_tensor.size(0) - 1)
    target_mean = torch.mean(target_tensor, dim=0)
    target_cov = torch.matmul((target_tensor - target_mean).t(), (target_tensor - target_mean)) / (
            target_tensor.size(0) - 1)

    # 计算KL散度
    kl_divergence = 0.5 * (torch.trace(torch.inverse(target_cov) @ source_cov) +
                           torch.trace(torch.inverse(source_cov) @ target_cov) +
                           torch.norm(target_mean - source_mean) ** 2 -
                           source_tensor.size(1))

    return kl_divergence.item()


def CORAL(source, target, **kwargs):
    # 获取源域和目标域的数据维度
    d = source.data.shape[1]
    # 获取源域和目标域的样本数量
    ns, nt = source.data.shape[0], target.data.shape[0]
    # source=torch.log(source)
    # target=torch.log(target)
    # 计算源域的协方差矩阵
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm / (ns - 1)

    # 计算目标域的协方差矩阵
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt / (nt - 1)

    # 计算源域和目标域之间的Frobenius范数
    loss = torch.mul((xc - xct), (xc - xct))
    loss = torch.sum(loss) / (4 * d * d)

    return loss


def guassian_kernel(source, target, kernel_mul=2, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并
    # 将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0 - total1) ** 2).sum(2)
    # 调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    # 高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # 得到最终的核矩阵
    return sum(kernel_val)  # /len(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # 根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss  # 因为一般都是n==m，所以L矩阵一般不加入计算


def main(args: argparse.Namespace):
    print(args)

    cudnn.benchmark = True
    device = args.device

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)

    # source_clean_dataset = ImageList(open(args.src_address.split('.t')[0] + '_true.txt').readlines(),
    #                                  transform=train_transform)
    # source_noise_dataset = ImageList(open(args.src_address.split('.t')[0] + '_false.txt').readlines(),
    #                                  transform=train_transform)
    # source_dataset = ConcatDataset([source_clean_dataset, source_noise_dataset])
    source_dataset = ImageList(open(args.src_address).readlines(), transform=train_transform)
    target_dataset = ImageList(open(args.tgt_address).readlines(), transform=val_transform)

    source_loader = DataLoader(source_dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.workers, drop_last=True)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.workers, drop_last=True)

    source_iter = ForeverDataIterator(source_loader)
    target_iter = ForeverDataIterator(target_loader)

    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)
    classifier_feature_dim = classifier.features_dim

    if args.randomized:
        domain_discri = DomainDiscriminator(
            args.randomized_dim, hidden_size=1024).to(device)
    else:
        domain_discri = DomainDiscriminator(
            classifier_feature_dim * num_classes, hidden_size=1024).to(device)

    all_parameters = classifier.get_parameters() + domain_discri.get_parameters()
    # define optimizer and lr scheduler
    optimizer = SGD(all_parameters, args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay, nesterov=True)
    t_total = args.iters_per_epoch * args.epochs
    print("{INFORMATION} The total number of steps is ", t_total)

    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr *
                                                 (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    domain_adv = ConditionalDomainAdversarialLoss(
        domain_discri, entropy_conditioning=args.entropy,
        num_classes=num_classes, features_dim=classifier_feature_dim, randomized=args.randomized,
        randomized_dim=args.randomized_dim
    ).to(device)

    # start training
    temp1 = [2.7]  # 2.7
    temp2 = [8.05556]  # 7
    for a in temp1:
        for b in temp2:
            best_acc1 = 0.
            for epoch in range(args.epochs):
                # train for one epoch
                print('epoch', epoch)
                train(source_iter, target_iter, classifier, domain_adv, optimizer,
                      lr_scheduler, epoch, args, a, b)

                # evaluate on validation set
                acc1 = utils.validate(target_loader, classifier, args, device)
                best_acc1 = max(acc1, best_acc1)
                if acc1 == best_acc1:
                    model = classifier

            model.train()
            f_s = torch.tensor([]).to(device).detach().cpu()
            f_t = torch.tensor([]).to(device).detach().cpu()
            for i, (images, source) in enumerate(source_loader):
                images = images.to(device)
                source = source.to(device)
                # compute output
                y, f1 = model(images)
                f_s = torch.cat((f_s, f1.detach().cpu()), dim=0)
            for i, (images, target) in enumerate(target_loader):
                images = images.to(device)
                target = target.to(device)
                # compute output
                y, f2 = model(images)
                f_t = torch.cat((f_t, f2.detach().cpu()), dim=0)
            tsne.visualize(f_s.detach().cpu(), f_t.detach().cpu(), 'a2w.png')
            a_distance.calculate(f_s.detach().cpu(), f_t.detach().cpu(), device)

            with open("accx.txt", "a+") as file:
                file.write(str(a) + "   ," + str(b) + "   ," + str(best_acc1) + "  time3\n")  # 将 best_acc 转换为字符串并写入文件
            print("best_acc2 = {:4.2f}".format(best_acc1))

    # with torch.no_grad():
    #     outS, outSL = net(_Xs)
    #     outT, outTL = net(_Xt)
    #     f_s = outS.cpu().numpy()
    #     f_t = outT.cpu().numpy()

    # np.savetxt('DW_source.csv', f_s , delimiter = ',')
    # np.savetxt('DW_target.csv', f_t, delimiter = ',')


def train(source_iter, target_iter, model: ImageClassifier,
          domain_adv: ConditionalDomainAdversarialLoss, optimizer: SGD,
          lr_scheduler, epoch: int, args: argparse.Namespace, a, b):
    c = 5.04e3  # 5.04e3
    d = 1.8  # 1.8
    print(a, b, c, d)
    device = args.device
    # switch to train mode
    model.train()
    domain_adv.train()
    # max_loss=0
    for i in range(args.iters_per_epoch):
        # print('iter', i)
        x_s, labels_s, = next(source_iter)
        x_t, _ = next(target_iter)
        x_s = x_s.to(device)
        x_t = x_t.to(device)

        # labels_s = torch.tensor([int(element) for element in labels_s])
        labels_s = labels_s.to(device)
        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        # domain_discri = DomainDiscriminator(
        #   256, hidden_size=1024).to(device)
        # label_pre=find_max_indices(y_s)
        # acc=calculate_accuracy(label_pre,labels_s)
        # total_acc+=acc
        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss, f_s_fft = domain_adv(y_s, f_s, y_t, f_t)
        # print(y_s)
        # print(labels_s)
        # print(len(y_s))
        # fw_s=domain_discri(f_s)
        # fw_fft=domain_discri(f_s_fft)
        # fw_s=torch.mean(fw_s)
        # fw_fft=torch.mean(fw_fft)
        # print(fw_s,fw_fft)
        # mmd_loss=mmd_rbf(f_s,f_s_fft)
        # mmd_loss2=mmd_rbf(f_s,f_t)
        # mmd_loss2=mmd_rbf(f_s,f_s_fft)
        # # print('cls_loss', cls_loss)
        # # print('ot_loss', ot_loss)
        # KL_loss=compute_kl_divergence(f_s, f_s_fft)
        # print(KL_loss*c)
        # print(KL_loss)
        # max_loss=max(max_loss,KL_loss)
        coral_loss = CORAL(f_s, f_s_fft)
        # print("@@@@@@@@@@@@@@@@@")
        # print(CORAL(f_s,f_s_fft)*5e3)
        # print("#########")
        # coral_loss2 = CORAL(f_s,f_t)i
        # print(compute_lmmd(f_s,f_s_fft)*c)
        # print(f_s.size())
        # wd_loss=torch.abs(fw_s-fw_fft)
        # M = ot.dist(f_s.detach().cpu().numpy(),f_s_fft.detach().cpu().numpy())  # 计算成本矩阵
        # emd = ot.emd2([], [], M)
        # N = ot.dist(f_s.detach().cpu().numpy(),f_t.detach().cpu().numpy())  # 计算成本矩阵
        # emd2 = ot.emd2([], [], N)
        # print(emd+emd2)
        # print(emd)
        # ot_loss=OT(y_s,f_s,y_s,f_s_fft,labels_s)
        # ot_loss2=OT(y_s,f_s,y_t,f_t,labels_s)
        loss = d * a * cls_loss + d * transfer_loss * b + d * coral_loss * c
        # print(a * cls_loss)
        # print(transfer_loss * b)
        # print(coral_loss)
        # print(wd_loss)
        # print("#########")
        # print(compute_lmmd(f_s,f_s_fft))
        # print(compute_lmmd(f_s,f_t))
        # print(coral_loss,coral_loss2)
        # print("############")
        # print(KL_loss)
        # cls_acc = accuracy(y_s, labels_s)[0]

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
    # tsne.visualize(f_s.detach().cpu(),f_t.detach().cpu(),'test.png')
    # print(max_loss)


def OT(ys, xs, yt, xt, labels_s):
    C0 = cdist(xs.cpu().detach().numpy(), xt.cpu().detach().numpy(), metric='sqeuclidean')
    C1 = cdist(ys.cpu().detach().numpy(), yt.cpu().detach().numpy(), metric='sqeuclidean')  # labels_s
    C = 10 * C0 + C1

    OUTS = ot.unif(xs.shape[0])
    OUTT = ot.unif(xt.shape[0])
    gamma = ot.emd(OUTS, OUTT, C)

    ## update gamma
    # parameters
    W = numpy.zeros((args.batch_size, args.batch_size))
    mu = 10000  # this one can be tuned 1000 100000
    rho = 6  # this one can be tuned 1.2   6
    # lam = 10

    for num in range(1, 2):
        temp = gamma + W / mu
        temp = temp.astype(float)
        # print(temp.dtype)
        U, S, V = numpy.linalg.svd(temp, 'econ')
        V = V.T

        diagS = S
        svp = len(numpy.flatnonzero(diagS > 1.0 / mu))
        diagS = numpy.maximum(0, diagS - 1.0 / mu)

        if svp < 0.5:  # svp = 0
            svp = 1

        # if svp1>=1:
        #     diagS = diagS[1:svp]-1.0 / mu
        # else:
        #     svp=1
        #     diagS = 0

        # print('S', S)
        # print('diagS', diagS)

        J_hat = U[:, 0:svp].dot(numpy.diag(diagS[0:svp]).dot(V[:, 0:svp].T))
        # J_hat = U[:, 1:svp].dot(np.diag(diagS).dot(V[:, 1:svp].T))
        # a1 = J_hat

        gamma = (J_hat - W - 0.001 * numpy.diag(numpy.diag(C)) / mu)  # 0.001
    #     # print('gamma', gamma)
    #
    #     H2 = gamma - J_hat
    #     W = W + mu * H2
    #     mu = rho * mu
    #
    #     norm = numpy.linalg.norm(H2, 'fro')

    gamma = FN(gamma).to(device)
    C = torch.tensor(C, requires_grad=True).to(device)
    OT_loss = 0.001 * torch.sum(gamma * C)

    return OT_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Low-rank Optimal Transport for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('--dataset', default='Office-home', type=str,
                        help='which dataset')
    parser.add_argument('--src_address', default=r'/root/autodl-fs/LROT_supp/data/Office-31/webcam.txt',
                        type=str,
                        help='address of image list of source dataset')
    parser.add_argument('--tgt_address', default=r'/root/autodl-fs/LROT_supp/data/Office-31/amazon.txt', type=str,
                        help='address of image list of target dataset')
    parser.add_argument('--stats_file', default=None, type=str,
                        help='store the training loss and and validation acc')
    parser.add_argument('--noisy_rate', default=0.4, type=float,
                        help='noisy rate')

    parser.add_argument('--del_rate', default=0.2, type=float,
                        help='delete rate of sample for transfer')

    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('--arch', metavar='ARCH', default='resnet50',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true',
                        help='whether train from scratch.')
    parser.add_argument('--entropy', default=False,
                        action='store_true', help='use entropy conditioning')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--randomized', action='store_true',
                        help='using randomized multi-linear-map (default: False)')
    parser.add_argument('--randomized-dim', default=1024, type=int,
                        help='randomized dimension when using randomized multi-linear-map (default: 1024)')
    # training parameters
    parser.add_argument('--batch-size', default=24, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')  # 0.002
    parser.add_argument('--lr-gamma', default=0.001,
                        type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75,
                        type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9,
                        type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('--iters-per-epoch', default=20, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--gpu', type=str, default="0", help="GPU ID")
    parser.add_argument('--seed', default=7, type=int,
                        help='seed for initializing training. ')

    args = parser.parse_args()
    # torch.cuda.device_count()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # torch.cuda.set_device(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    args.device = device

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # cudnn.deterministic = True

    if args.dataset == 'Office-31':
        num_classes = 31
    elif args.dataset == 'Office-home':
        num_classes = 65
    else:
        width = -1

    main(args)
