"""
Compute
Inception Score (IS),
Frechet Inception Discrepency (FID), ref "https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py"
Maximum Mean Discrepancy (MMD)
for a set of fake images

use numpy array
Xr: high-level features for real images; nr by d array
Yr: labels for real images
Xg: high-level features for fake images; ng by d array
Yg: labels for fake images
IMGSr: real images
IMGSg: fake images

"""

"""
总结
•FID： 评价分布一致性
•Label Score： 评价条件一致性
•Inception Score： 评价图片质量,多样性
"""

import gc

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
# from numpy import linalg as LA
from scipy import linalg
from scipy.stats import entropy
from torch.autograd import Variable
from torch.nn import functional as F

from .utils import SimpleProgressBar, ImgsDataset


# 归一化输入图像到 [-1, 1] 范围
# 很多预训练网络（例如 Inception、DCGAN 中的判别器等）要求输入图像在此范围内。
def normalize_images(batch_images):
    batch_images = batch_images / 255.0  # ->[0,1]
    batch_images = (batch_images - 0.5) / 0.5  # -> [-1,1]
    return batch_images


##############################################################################
# FID scores
##############################################################################
# compute FID based on extracted features
def FID(Xr, Xg, eps=1e-10):
    """
    计算两个高斯分布之间的 Frechet 距离。假设
         Xr ~ N(mu_r, sigma_r)  和  Xg ~ N(mu_g, sigma_g)
    则 Frechet 距离为：
         ||mu_r - mu_g||^2 + Trace(sigma_r + sigma_g - 2*sqrt(sigma_r*sigma_g))
    """
    # sample mean
    MUr = np.mean(Xr, axis=0)
    MUg = np.mean(Xg, axis=0)
    mean_diff = MUr - MUg

    # sample covariance
    # 协方差矩阵（注意 np.cov 默认以每行代表一个变量，因此需转置）
    SIGMAr = np.cov(Xr.transpose())
    SIGMAg = np.cov(Xg.transpose())

    # 计算协方差矩阵乘积的矩阵平方根
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(SIGMAr.dot(SIGMAg), disp=False)  # square root of a matrix
    covmean = covmean.real  # 可能会有少量虚部，取实部即可

    # 如果乘积几乎奇异，加入微小偏置后重新计算
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(SIGMAr.shape[0]) * eps
        covmean = linalg.sqrtm((SIGMAr + offset).dot(SIGMAg + offset))

    # fid score
    fid_score = mean_diff.dot(mean_diff) + np.trace(SIGMAr + SIGMAg - 2 * covmean)

    return fid_score


##test
# Xr = np.random.rand(10000,1000)
# Xg = np.random.rand(10000,1000)
# print(FID(Xr, Xg))

# compute FID from raw images
def cal_FID(PreNetFID, IMGSr, IMGSg, batch_size=500, resize=None, norm_img=False):
    """
    从原始图像提取特征并计算 FID
    :param PreNetFID: PreNetFID 为预训练网络，用于提取特征
    """
    # resize: if None, do not resize; if resize = (H,W), resize images to 3 x H x W

    PreNetFID.eval()

    nr = IMGSr.shape[0]
    ng = IMGSg.shape[0]

    nc = IMGSr.shape[1]  # IMGSr is nrxNCxIMG_SIExIMG_SIZE
    img_size = IMGSr.shape[2]

    if batch_size > min(nr, ng):
        batch_size = min(nr, ng)
        # print("FID: recude batch size to {}".format(batch_size))

    # compute the length of extracted features
    with torch.no_grad():
        test_img = torch.from_numpy(IMGSr[0].reshape((1, nc, img_size, img_size))).type(
                torch.float).cuda()
        if resize is not None:
            test_img = nn.functional.interpolate(test_img, size=resize, scale_factor=None,
                                                 mode='bilinear', align_corners=False)
        if norm_img:
            test_img = normalize_images(test_img)
        # _, test_features = PreNetFID(test_img)
        test_features = PreNetFID(test_img)
        d = test_features.shape[1]  # length of extracted features

    Xr = np.zeros((nr, d))
    Xg = np.zeros((ng, d))

    # batch_size = 500
    with torch.no_grad():
        tmp = 0
        pb1 = SimpleProgressBar()
        for i in range(nr // batch_size):
            imgr_tensor = torch.from_numpy(IMGSr[tmp:(tmp + batch_size)]).type(torch.float).cuda()
            if resize is not None:
                imgr_tensor = nn.functional.interpolate(imgr_tensor, size=resize, scale_factor=None,
                                                        mode='bilinear', align_corners=False)
            if norm_img:
                imgr_tensor = normalize_images(imgr_tensor)
            # _, Xr_tmp = PreNetFID(imgr_tensor)
            Xr_tmp = PreNetFID(imgr_tensor)
            Xr[tmp:(tmp + batch_size)] = Xr_tmp.detach().cpu().numpy()
            tmp += batch_size
            # pb1.update(min(float(i)*100/(nr//batch_size), 100))
            pb1.update(min(max(tmp / nr * 100, 100), 100))
        del Xr_tmp, imgr_tensor;
        gc.collect()
        torch.cuda.empty_cache()

        tmp = 0
        pb2 = SimpleProgressBar()
        for j in range(ng // batch_size):
            imgg_tensor = torch.from_numpy(IMGSg[tmp:(tmp + batch_size)]).type(torch.float).cuda()
            if resize is not None:
                imgg_tensor = nn.functional.interpolate(imgg_tensor, size=resize, scale_factor=None,
                                                        mode='bilinear', align_corners=False)
            if norm_img:
                imgg_tensor = normalize_images(imgg_tensor)
            # _, Xg_tmp = PreNetFID(imgg_tensor)
            Xg_tmp = PreNetFID(imgg_tensor)
            Xg[tmp:(tmp + batch_size)] = Xg_tmp.detach().cpu().numpy()
            tmp += batch_size
            # pb2.update(min(float(j)*100/(ng//batch_size), 100))
            pb2.update(min(max(tmp / ng * 100, 100), 100))
        del Xg_tmp, imgg_tensor;
        gc.collect()
        torch.cuda.empty_cache()

    fid_score = FID(Xr, Xg, eps=1e-6)

    return fid_score


##############################################################################
# label_score
# difference between assigned label and predicted label
##############################################################################
def cal_labelscore(PreNet, images, labels_assi, min_label_before_shift, max_label_after_shift,
        batch_size=200, resize=None, norm_img=False, num_workers=0):
    """
    该函数用于计算生成图像的标签评分，即衡量图像生成是否符合给定的条件标签。
    :param PreNet：预训练的分类网络
    :param images：生成图像
    :param labels_assi：生成时指定的标签
    :param min_label_before_shift, 用于对标签进行缩放转换的参数
    :param max_label_after_shift：用于对标签进行缩放转换的参数
    """

    PreNet.eval()

    # assume images are nxncximg_sizeximg_size
    n = images.shape[0]
    nc = images.shape[1]  # number of channels
    img_size = images.shape[2]
    labels_assi = labels_assi.reshape(-1)

    eval_trainset = ImgsDataset(images, labels_assi, normalize=False)
    eval_dataloader = torch.utils.data.DataLoader(eval_trainset, batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers)

    labels_pred = np.zeros(n + batch_size)

    nimgs_got = 0
    pb = SimpleProgressBar()
    for batch_idx, (batch_images, batch_labels) in enumerate(eval_dataloader):
        batch_images = batch_images.type(torch.float).cuda()
        batch_labels = batch_labels.type(torch.float).cuda()
        batch_size_curr = len(batch_labels)

        if norm_img:
            batch_images = normalize_images(batch_images)

        batch_labels_pred, _ = PreNet(batch_images)
        labels_pred[
        nimgs_got:(nimgs_got + batch_size_curr)] = batch_labels_pred.detach().cpu().numpy().reshape(
                -1)

        nimgs_got += batch_size_curr
        pb.update((float(nimgs_got) / n) * 100)

        del batch_images;
        gc.collect()
        torch.cuda.empty_cache()
    # end for batch_idx

    labels_pred = labels_pred[0:n]

    labels_pred = (labels_pred * max_label_after_shift) - np.abs(min_label_before_shift)
    labels_assi = (labels_assi * max_label_after_shift) - np.abs(min_label_before_shift)

    ls_mean = np.mean(np.abs(labels_pred - labels_assi))
    ls_std = np.std(np.abs(labels_pred - labels_assi))

    return ls_mean, ls_std


##############################################################################
# Compute Inception Score
##############################################################################
def inception_score(imgs, num_classes, net, cuda=True, batch_size=32, splits=1,
        normalize_img=False):
    """计算生成图像的 Inception Score
    :param imgs: 未归一化的 (3 x H x W) numpy 格式的图像数组
    :param num_classes: 分类数目（例如 Inception 网络通常在 ImageNet 上有 1000 类）
    :param net: 预训练的分类 CNN（如 Inception v3）
    :param cuda: 是否使用 GPU
    :param batch_size: 批处理大小
    :param splits: 将数据集分成几部分计算 score，用于统计分数的均值和标准差
    :param normalize_img: 是否先对图像进行归一化
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataset = ImgsDataset(imgs, labels=None, normalize=normalize_img)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Load inception model
    if cuda:
        net = net.cuda()
    else:
        net = net.cpu()
    net.eval();

    def get_pred(x):
        x, _ = net(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, num_classes))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)
