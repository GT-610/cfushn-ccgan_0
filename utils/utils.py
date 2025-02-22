"""
Some helpful functions

"""
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch


def fn_norm_labels(labels, max_label):
    """
    将未归一化的标签转换到 [0,1] 区间

    参数:
        labels (np.ndarray): 原始标签数组

    返回:
        np.ndarray: 归一化后的标签数组（除以 args.max_label）
    """
    return labels / max_label


def fn_denorm_labels(labels, max_label):
    """
    将归一化的标签还原为原始尺度

    参数:
        labels (np.ndarray 或 torch.Tensor 或数字): 归一化后的标签

    返回:
        与输入类型对应的标签，数值范围恢复到 [0, max_label]
    """
    if isinstance(labels, np.ndarray):
        return (labels * max_label).astype(int)
    elif torch.is_tensor(labels):
        return (labels * max_label).type(torch.int)
    else:
        return int(labels * max_label)


def hflip_images(batch_images):
    """ 对图像进行水平翻转（随机选择部分样本翻转） """
    uniform_threshold = np.random.uniform(0, 1, len(batch_images))
    index_gt = np.nonzero(uniform_threshold > 0.5)[0]
    batch_images[index_gt] = np.flip(batch_images[index_gt], axis=3)
    return batch_images


def normalize_images(batch_images):
    """将图像归一化到 [-1,1]（用于 GAN 训练）"""
    batch_images = batch_images / 255.0
    batch_images = (batch_images - 0.5) / 0.5
    return batch_images


# Progress Bar
class SimpleProgressBar:
    def __init__(self, width=50):
        self.last_x = -1
        self.width = width

    def update(self, x):
        assert 0 <= x <= 100  # `x`: progress in percent ( between 0 and 100)
        if self.last_x == int(x): return
        self.last_x = int(x)
        pointer = int(self.width * (x / 100.0))
        sys.stdout.write('\r%d%% [%s]' % (int(x), '#' * pointer + '.' * (self.width - pointer)))
        sys.stdout.flush()
        if x == 100:
            print('')


# torch dataset from numpy array
class ImgsDataset(torch.utils.data.Dataset):
    def __init__(self, images, cont_labels=None, class_labels=None, normalize=False):
        """
        初始化图像数据集，用于返回图像以及对应的标签信息。

        参数:
            images (np.ndarray): 图像数组，长度为样本数。
            cont_labels (np.ndarray, optional): 连续标签数组（例如年龄），形状与 images 长度相同。默认 None。
            class_labels (np.ndarray, optional): 离散标签数组（例如人种类别），形状与 images 长度相同。默认 None。
            normalize (bool, optional): 是否对图像进行归一化（将像素值归一化到 [-1,1]）。默认 False。
        """
        super(ImgsDataset, self).__init__()
        self.images = images
        self.n_images = len(self.images)
        self.cont_labels = cont_labels
        self.class_labels = class_labels
        # 如果存在任一标签，检查长度是否匹配
        if cont_labels is not None:
            if len(self.images) != len(self.cont_labels):
                raise Exception('images (' + str(len(self.images)) +
                                ') and cont_labels (' + str(len(self.cont_labels)) +
                                ') do not have the same length!!!')
        if class_labels is not None:
            if len(self.images) != len(self.class_labels):
                raise Exception('images (' + str(len(self.images)) +
                                ') and class_labels (' + str(len(self.class_labels)) +
                                ') do not have the same length!!!')
        self.normalize = normalize

    def __getitem__(self, index):
        """
        根据索引返回图像以及对应的标签信息。

        参数:
            index (int): 样本索引。

        返回:
            如果同时提供了连续标签和离散标签，则返回 (image, cont_label, class_label)；
            如果只提供连续标签，则返回 (image, cont_label)；
            如果只提供离散标签，则返回 (image, class_label)；
            如果都未提供，则仅返回 image。
        """
        image = self.images[index]
        if self.normalize:
            image = image / 255.0
            image = (image - 0.5) / 0.5

        # 根据是否有连续标签和离散标签返回不同结果
        if self.cont_labels is not None and self.class_labels is not None:
            cont_label = self.cont_labels[index]
            class_label = self.class_labels[index]
            return image, cont_label, class_label
        elif self.cont_labels is not None:
            return image, self.cont_labels[index]
        elif self.class_labels is not None:
            return image, self.class_labels[index]
        else:
            return image

    def __len__(self):
        """返回数据集中图像的总数。"""
        return self.n_images


def PlotLoss(loss, filename):
    x_axis = np.arange(start=1, stop=len(loss) + 1)
    plt.switch_backend('agg')
    mpl.style.use('seaborn')
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x_axis, np.array(loss))
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.legend()
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),  shadow=True, ncol=3)
    # plt.title('Training Loss')
    plt.savefig(filename)


# compute entropy of class labels; labels is a numpy array
# 计算类别分布的熵，输入 labels 为 numpy 数组
def compute_entropy(labels, base=None):
    _, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    base = np.e if base is None else base
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()


# 利用分类网络预测图像的类别标签
def predict_class_labels(net, images, batch_size=500, verbose=False, num_workers=0):
    net.eval()

    n = len(images)
    if batch_size > n:
        batch_size = n

    # 由于这里只进行预测，不需要标签，因此调用 ImgsDataset 时只传 images 参数
    dataset_pred = ImgsDataset(images, normalize=False)
    dataloader_pred = torch.utils.data.DataLoader(dataset_pred, batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers)
    # 用于存放预测的类别标签，预先分配空间
    class_labels_pred = np.zeros(n + batch_size)
    with torch.no_grad():
        nimgs_got = 0
        if verbose:
            pb = SimpleProgressBar()
        for batch_idx, batch_images in enumerate(dataloader_pred):
            batch_images = batch_images.type(torch.float).to(net.device)
            batch_size_curr = len(batch_images)
            # 这里调用 net(batch_images) 得到的是一个三元组，取第二个元素 y_class
            _, y_class, _ = net(batch_images)
            # 对 y_class（预测的类别 logits）计算 softmax 后取最大值的索引
            _, batch_class_labels_pred = torch.max(y_class.data, 1)
            # 将预测结果保存到数组中
            class_labels_pred[nimgs_got:(
                    nimgs_got + batch_size_curr)] = batch_class_labels_pred.detach().cpu().numpy().reshape(
                    -1)
            nimgs_got += batch_size_curr
            if verbose:
                pb.update(min((float(nimgs_got) / n) * 100, 100))
    class_labels_pred = class_labels_pred[0:n]
    return class_labels_pred
