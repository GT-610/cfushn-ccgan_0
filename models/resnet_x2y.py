"""
model_y2h.py
---------------
该文件定义了基于 ResNet 的模型，用于将输入图像从像素空间映射到特征嵌入空间。
模型需要在目标数据集上进行预训练，生成的嵌入特征可用于后续任务（例如密度比估计）。
如果 isometric_map = True，则会在最后增加一步，将特征映射的维度从 512 扩展到 32*32*3，
用于特征空间中的密度比估计。

参考文献:
    Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg

device = cfg.device


class BasicBlock(nn.Module):
    """
    ResNet 的基础残差模块 (BasicBlock)

    该模块由两个 3x3 卷积层构成，前面接 BatchNorm 和 ReLU，
    同时具有一个短接分支 (shortcut)，当输入和输出尺寸不匹配时，
    通过1x1卷积调整尺寸。
    """
    expansion = 1  # 输出通道数相对于基本通道数的扩展倍数（对BasicBlock为1）

    def __init__(self, in_planes, planes, stride=1):
        """
        Args:
            in_planes (int): 输入特征图的通道数
            planes (int): 基础卷积层的输出通道数
            stride (int, optional): 第一个卷积层的步幅，默认1；当stride不为1时，同时缩小空间尺寸。
        """
        super(BasicBlock, self).__init__()
        # 第一个卷积层，卷积核大小为3x3，padding=1保持尺寸，stride可调
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 第二个卷积层，步幅固定为1
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 如果输入输出尺寸不匹配，构建短接分支以调整尺寸
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride,
                              bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 加上短接分支的输出
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    ResNet 的瓶颈模块 (Bottleneck)

    该模块由1x1、3x3、1x1三层卷积构成，其中1x1卷积用于降维和升维，
    最终输出通道数为 planes * expansion。瓶颈设计用于更深层的网络。
    """
    expansion = 4  # 输出通道数相对于基础通道数的扩展倍数（对Bottleneck为4）

    def __init__(self, in_planes, planes, stride=1):
        """
        Args:
            in_planes (int): 输入特征图的通道数
            planes (int): 中间卷积层的通道数（降维后再升维）
            stride (int, optional): 第二层卷积的步幅，用于控制空间尺寸缩小，默认1
        """
        super(Bottleneck, self).__init__()
        # 第一层1x1卷积，用于降维
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 第二层3x3卷积，stride可能不为1
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 第三层1x1卷积，用于升维
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        # 如果输入输出尺寸不匹配，构建短接分支调整尺寸
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride,
                              bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetX2Y(nn.Module):
    """
    基于 ResNet 的图像嵌入网络

    该模型通过一系列卷积层和残差块提取图像特征，并将特征映射到预设的嵌入空间。
    最后，通过全连接层将嵌入特征映射到一个标量输出（例如用于回归或密度估计）。

    参数说明：
        block (class): 残差块类型，可选 BasicBlock 或 Bottleneck
        num_blocks (list of int): 每个残差层中包含的残差块数量，例如 [2,2,2,2] 表示 ResNet18
        nc (int, optional): 输入图像的通道数，默认NC（3）
        dim_embed (int, optional): 嵌入空间的维度，默认 DIM_EMBED (128)
        ngpu (int, optional): 使用的 GPU 数量，默认1；若 ngpu > 1，则使用数据并行
    """

    def __init__(self, block, num_blocks, nc=3, dim_embed=cfg.dim_embed, ngpu=1):
        super(ResNetX2Y, self).__init__()
        self.in_planes = 64  # 初始卷积层输出通道数
        self.ngpu = ngpu  # GPU 数量
        # todo 并行处理这块没有仔细研究

        # 主干网络，包含初始卷积层、BatchNorm、ReLU和多个残差层
        self.main = nn.Sequential(
                # 初始卷积层，将输入 nc 通道映射到 64 通道，kernel_size=3，padding=1保持尺寸
                nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # 依次构建4个残差层，每个残差层通过 _make_layer 构建
                # 注意：这里部分层采用 stride=2，实现空间尺寸缩小
                self._make_layer(block, 64, num_blocks[0], stride=2),  # 输出尺寸：原始尺寸/2
                self._make_layer(block, 128, num_blocks[1], stride=2),  # 输出尺寸：原始尺寸/4
                self._make_layer(block, 256, num_blocks[2], stride=2),  # 输出尺寸：原始尺寸/8
                self._make_layer(block, 512, num_blocks[3], stride=2),  # 输出尺寸：原始尺寸/16
                # 自适应平均池化，将特征图池化为 1x1 尺寸
                nn.AdaptiveAvgPool2d((1, 1))
        )

        # 全连接层，将提取的 512 维特征进一步映射到嵌入空间
        self.x2h_res = nn.Sequential(
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),

                nn.Linear(512, dim_embed),
                nn.BatchNorm1d(dim_embed),
                nn.ReLU(),
        )

        # 最后输出层,分支1: h2y_cont，将嵌入特征映射到一个标量输出(用于回归标签)
        self.h2y_cont = nn.Sequential(
                nn.Linear(dim_embed, 1),
                nn.ReLU()
                # 这里默认使用非负回归标签, 训练时候使用均方误差
        )
        # 最后输出层,分支2: h2y_class，将嵌入特征映射到一个标量输出(用于离散标签）
        self.h2y_class = nn.Sequential(
                nn.Linear(dim_embed, cfg.num_classes),
                # 这个分支输出的本质是分类, 最后输出保留logits用于训练时候计算交叉熵损失
                # 故不需要激活函数，因为后续通常使用 CrossEntropyLoss，该损失函数内部会做 softmax 处理
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        构建由多个残差块组成的层

        Args:
            block (class): 残差块类型（BasicBlock 或 Bottleneck）
            planes (int): 基础卷积层的输出通道数
            num_blocks (int): 当前层包含的残差块数量
            stride (int): 第一个残差块的步幅，控制空间尺寸缩小

        Returns:
            nn.Sequential: 由多个残差块按顺序组成的层
        """
        # 第一个块使用给定 stride，其余块步幅均为1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion  # 更新 in_planes 以匹配下一层输入
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播过程

        Args:
            x (torch.Tensor): 输入图像，形状为 (batch_size, nc, H, W)

        Returns:
            tuple: (out, features)
                - out (torch.Tensor): 最终输出的标量结果，形状为 (batch_size, 1)
                - features (torch.Tensor): 从 x2h_res 层输出的嵌入特征，形状为 (batch_size, dim_embed)
        """
        # 若使用多GPU，则利用数据并行
        if x.is_cuda and self.ngpu > 1:
            features = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
            features = features.view(features.size(0), -1)  # 展平为 (batch_size, 512)
            features = nn.parallel.data_parallel(self.x2h_res, features, range(self.ngpu))
            # out = nn.parallel.data_parallel(self.h2y, features, range(self.ngpu))
            y_cont = nn.parallel.data_parallel(self.h2y_cont, features, range(self.ngpu))
            y_class = nn.parallel.data_parallel(self.h2y_class, features, range(self.ngpu))
        else:
            features = self.main(x)
            features = features.view(features.size(0), -1)
            features = self.x2h_res(features)
            # out = self.h2y(features)
            y_cont = self.h2y_cont(features)
            y_class = self.h2y_class(features)
        return y_cont, y_class, features


def ResNet18_x2y(dim_embed=cfg.dim_embed, ngpu=1):
    return ResNetX2Y(BasicBlock, [2, 2, 2, 2], dim_embed=dim_embed, ngpu=ngpu)


def ResNet34_x2y(dim_embed=cfg.dim_embed, ngpu=1):
    return ResNetX2Y(BasicBlock, [3, 4, 6, 3], dim_embed=dim_embed, ngpu=ngpu)


def ResNet50_x2y(dim_embed=cfg.dim_embed, ngpu=1):
    return ResNetX2Y(Bottleneck, [3, 4, 6, 3], dim_embed=dim_embed, ngpu=ngpu)


if __name__ == "__main__":
    net = ResNet34_x2y(ngpu=1).to(device)
    x = torch.randn(16, 3, cfg.img_size, cfg.img_size).to(device)  # 随机生成 16 张 64x64 RGB 图像
    out, features = net(x)
    print("输出标量尺寸:", out.size())
    print("嵌入特征尺寸:", features.size())
