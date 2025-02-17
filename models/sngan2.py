import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm  # 用于给层施加谱归一化，控制 Lipschitz 常数

from config.config import device

# 全局设置：图像通道数和是否使用偏置项
channels = 3
bias = True


######################################################################################################################
# Generator 部分
######################################################################################################################

# 条件批归一化层：根据输入条件对 BN 层的输出进行调制
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, dim_embed):
        """
        Args:
            num_features (int): 输入特征图的通道数
            dim_embed (int): 条件嵌入向量的维度
        """
        super().__init__()
        self.num_features = num_features
        # 不使用 affine 参数，由外部条件提供 gamma 和 beta
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        # 将条件嵌入映射到每个通道对应的缩放系数 gamma（无偏置）
        self.embed_gamma = nn.Linear(dim_embed, num_features, bias=False)
        # 将条件嵌入映射到每个通道对应的平移系数 beta（无偏置）
        self.embed_beta = nn.Linear(dim_embed, num_features, bias=False)

    def forward(self, x, y):
        """
        前向传播：对输入 x 进行 BN，然后根据条件 y 调制
        Args:
            x (Tensor): 特征图，形状 (N, num_features, H, W)
            y (Tensor): 条件嵌入向量，形状 (N, dim_embed)
        Returns:
            Tensor: 调制后的特征图
            out = BN(x) + BN(x) \times \gamma(y) + \beta(y)
        """
        out = self.bn(x)  # 对 x 进行批归一化（无 affine）
        # 将条件向量映射到 gamma，形状变为 (N, num_features)
        gamma = self.embed_gamma(y).view(-1, self.num_features, 1, 1)
        # 将条件向量映射到 beta
        beta = self.embed_beta(y).view(-1, self.num_features, 1, 1)
        # 这里采用的公式: out = out + out * gamma + beta
        # （相当于 out * (1 + gamma) + beta）
        out = out + out * gamma + beta
        return out


# 生成器中的残差块，支持条件和无条件情况
class ResBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, dim_embed, bias=True):
        """
        构建生成器中的一个残差块，可用于上采样
        Args:
            in_channels (int): 输入特征图的通道数
            out_channels (int): 输出特征图的通道数
            dim_embed (int): 条件嵌入向量的维度，用于条件批归一化
            bias (bool): 卷积层是否使用偏置项
        """
        super(ResBlockGenerator, self).__init__()

        # 定义两个 3x3 卷积层，保持空间尺寸不变（padding=1）
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=bias)
        # 使用 Xavier 均匀初始化卷积权重，激活函数为 ReLU 时 sqrt(2) 的放缩系数
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))

        # 条件批归一化层，分别作用于第一个卷积前和第二个卷积前
        self.condgn1 = ConditionalBatchNorm2d(in_channels, dim_embed)
        self.condgn2 = ConditionalBatchNorm2d(out_channels, dim_embed)
        # 定义激活函数
        self.relu = nn.ReLU()
        # 定义上采样层（尺度因子为2，使特征图尺寸加倍）
        self.upsample = nn.Upsample(scale_factor=2)

        # 以下部分为无条件情况构建的网络分支（通常不用，主要用于测试或备用）
        self.model = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),
                self.conv1,
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                self.conv2
        )

        # 定义旁路（skip connection）分支，用于保留输入信息
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0,
                                     bias=bias)  # 1x1 卷积调整通道数
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)
        self.bypass = nn.Sequential(
                nn.Upsample(scale_factor=2),  # 同样进行上采样
                self.bypass_conv,
        )

    def forward(self, x, y):
        """
        前向传播：
            如果提供条件 y，则使用条件 BN 模块对 x 进行调制、上采样、卷积等操作，
            最后与旁路分支相加；
            如果 y 为 None，则使用无条件分支。
        Args:
            x (Tensor): 输入特征图
            y (Tensor or None): 条件嵌入向量；当 y 不为 None 时，采用条件批归一化
        Returns:
            Tensor: 输出特征图
        """
        if y is not None:
            # 条件情况：先对 x 用条件 BN 调制，再 ReLU 激活
            out = self.condgn1(x, y)
            out = self.relu(out)
            # 上采样，将空间尺寸扩大一倍
            out = self.upsample(out)
            # 第一层卷积转换通道数
            out = self.conv1(out)
            # 第二个条件 BN 调制
            out = self.condgn2(out, y)
            out = self.relu(out)
            # 第二层卷积
            out = self.conv2(out)
            # 加上旁路分支（对原始 x 进行上采样和 1x1 卷积）
            out = out + self.bypass(x)
        else:
            # 无条件情况：直接使用预定义的模型分支
            out = self.model(x) + self.bypass(x)
        return out


# 生成器整体结构，基于 SNGAN 架构
class sngan_generator(nn.Module):
    def __init__(self, nz=256, dim_embed=128, gen_ch=64):
        """
        构造生成器模型
        Args:
            nz (int, optional): 噪声向量 z 的维度，默认 256
            dim_embed (int, optional): 条件嵌入向量的维度，默认 128
            gen_ch (int, optional): 生成器通道基数，决定各层通道数，默认 64
        """
        super(sngan_generator, self).__init__()
        self.z_dim = nz
        self.dim_embed = dim_embed
        self.gen_ch = gen_ch

        # 将噪声向量 z 通过全连接层映射为 4x4 尺寸的特征图，通道数为 gen_ch*16
        self.dense = nn.Linear(self.z_dim, 4 * 4 * gen_ch * 16, bias=True)
        # 最后通过卷积将通道数转换为图像通道数，并经过 Tanh 映射到 [-1,1]
        self.final = nn.Conv2d(gen_ch, channels, 3, stride=1, padding=1, bias=bias)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

        # 依次定义 4 个残差块，实现从 4x4 到 64x64 的上采样过程
        self.genblock0 = ResBlockGenerator(gen_ch * 16, gen_ch * 8,
                                           dim_embed=dim_embed)  # 4x4 -> 8x8
        self.genblock1 = ResBlockGenerator(gen_ch * 8, gen_ch * 4,
                                           dim_embed=dim_embed)  # 8x8 -> 16x16
        self.genblock2 = ResBlockGenerator(gen_ch * 4, gen_ch * 2,
                                           dim_embed=dim_embed)  # 16x16 -> 32x32
        self.genblock3 = ResBlockGenerator(gen_ch * 2, gen_ch,
                                           dim_embed=dim_embed)  # 32x32 -> 64x64

        # 最后对生成的 64x64 特征图进行 BN、ReLU、卷积和 Tanh 激活输出图像
        self.final = nn.Sequential(
                nn.BatchNorm2d(gen_ch),
                nn.ReLU(),
                self.final,
                nn.Tanh()
        )

    def forward(self, z, y):
        """
        前向传播：
            输入噪声 z 和条件嵌入 y（已在嵌入空间中），
            首先通过全连接层得到 4x4 特征图，
            然后依次通过 4 个残差块进行上采样，
            最后输出生成的图像。
        Args:
            z (Tensor): 噪声向量，形状 (N, z_dim)
            y (Tensor): 条件嵌入向量，形状 (N, dim_embed)
        Returns:
            Tensor: 生成的图像，形状 (N, channels, 64, 64)
        """
        z = z.view(z.size(0), z.size(1))
        out = self.dense(z)
        # 将 dense 输出重塑为 (N, gen_ch*16, 4, 4)
        out = out.view(-1, self.gen_ch * 16, 4, 4)

        # 依次通过各个生成器残差块，每个块都会上采样一次
        out = self.genblock0(out, y)
        out = self.genblock1(out, y)
        out = self.genblock2(out, y)
        out = self.genblock3(out, y)
        out = self.final(out)
        return out


######################################################################################################################
# Discriminator 部分
######################################################################################################################

# 判别器中使用的残差块
class ResBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        构建判别器中的残差块，用于特征提取，同时使用谱归一化保证训练稳定性
        Args:
            in_channels (int): 输入特征图的通道数
            out_channels (int): 输出特征图的通道数
            stride (int, optional): 卷积步幅，若 stride != 1 则同时执行下采样
        """
        super(ResBlockDiscriminator, self).__init__()
        # 定义两个 3x3 卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=bias)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))

        # 根据 stride 选择是否添加平均池化层实现下采样
        if stride == 1:
            self.model = nn.Sequential(
                    nn.ReLU(),
                    spectral_norm(self.conv1),
                    nn.ReLU(),
                    spectral_norm(self.conv2)
            )
        else:
            self.model = nn.Sequential(
                    nn.ReLU(),
                    spectral_norm(self.conv1),
                    nn.ReLU(),
                    spectral_norm(self.conv2),
                    nn.AvgPool2d(2, stride=stride, padding=0)
            )

        # 定义旁路分支：1x1 卷积调整通道数，再根据 stride 下采样
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0, bias=bias)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)
        if stride != 1:
            self.bypass = nn.Sequential(
                    spectral_norm(self.bypass_conv),
                    nn.AvgPool2d(2, stride=stride, padding=0)
            )
        else:
            self.bypass = nn.Sequential(
                    spectral_norm(self.bypass_conv)
            )

    def forward(self, x):
        """
        前向传播，将主分支与旁路分支相加
        Args:
            x (Tensor): 输入特征图
        Returns:
            Tensor: 残差块输出
        """
        return self.model(x) + self.bypass(x)


# 第一层判别器残差块，特殊设计以避免在对原始图像做卷积前激活
class FirstResBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        构造判别器第一层的残差块，直接对原始图像进行卷积处理，而不先进行激活操作。
        Args:
            in_channels (int): 输入图像通道数
            out_channels (int): 输出特征图的通道数
            stride (int, optional): 控制下采样步幅，通常为 1 或 2
        """
        super(FirstResBlockDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=bias)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0, bias=bias)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)

        # 模型分支：先进行卷积，再激活，最后平均池化实现下采样
        self.model = nn.Sequential(
                spectral_norm(self.conv1),
                nn.ReLU(),
                spectral_norm(self.conv2),
                nn.AvgPool2d(2)
        )
        # 旁路分支：先平均池化，再通过 1x1 卷积
        self.bypass = nn.Sequential(
                nn.AvgPool2d(2),
                spectral_norm(self.bypass_conv),
        )

    def forward(self, x):
        """
        前向传播：将主分支与旁路分支相加后输出
        """
        return self.model(x) + self.bypass(x)


# 判别器整体结构
class sngan_discriminator(nn.Module):
    def __init__(self, dim_embed=128, disc_ch=64):
        """
        构造 SNGAN 判别器，支持条件判别（连续条件嵌入）
        Args:
            dim_embed (int, optional): 条件嵌入向量的维度
            disc_ch (int, optional): 判别器基础通道数
        """
        super(sngan_discriminator, self).__init__()
        self.dim_embed = dim_embed
        self.disc_ch = disc_ch

        # 第一部分残差块，从原始图像（channels）转换到较高通道数，同时下采样
        self.discblock1 = nn.Sequential(
                FirstResBlockDiscriminator(channels, disc_ch, stride=2),  # 将 64x64 图像下采样到 32x32
                ResBlockDiscriminator(disc_ch, disc_ch * 2, stride=2),  # 32x32 -> 16x16
                ResBlockDiscriminator(disc_ch * 2, disc_ch * 4, stride=2),  # 16x16 -> 8x8
        )
        # 第二部分残差块，继续下采样
        self.discblock2 = ResBlockDiscriminator(disc_ch * 4, disc_ch * 8, stride=2)  # 8x8 -> 4x4
        # 第三部分残差块，保持尺寸不变
        self.discblock3 = nn.Sequential(
                ResBlockDiscriminator(disc_ch * 8, disc_ch * 16, stride=1),  # 4x4 -> 4x4
                nn.ReLU(),
        )

        # 全连接部分：将提取到的特征展平后经过线性层输出一个标量得分
        self.linear1 = nn.Linear(disc_ch * 16 * 4 * 4, 1, bias=True)
        nn.init.xavier_uniform_(self.linear1.weight.data, 1.)
        self.linear1 = spectral_norm(self.linear1)
        # 条件嵌入部分：将条件嵌入向量通过线性映射到与判别器中间特征尺寸相同的向量
        self.linear2 = nn.Linear(self.dim_embed, disc_ch * 16 * 4 * 4, bias=False)
        nn.init.xavier_uniform_(self.linear2.weight.data, 1.)
        self.linear2 = spectral_norm(self.linear2)

    def forward(self, x, y):
        """
        前向传播：
            对输入图像 x 经过多个残差块提取特征，然后展平，
            再利用两个线性层分别计算无条件分支得分和条件内积项，
            最后将二者相加得到最终的判别得分。
        Args:
            x (Tensor): 输入图像，形状 (N, channels, H, W)
            y (Tensor): 条件嵌入向量，形状 (N, dim_embed)
        Returns:
            Tensor: 判别器输出得分，形状 (N, 1)
        """
        output = self.discblock1(x)  # 第一个残差块组
        output = self.discblock2(output)  # 第二个残差块组
        output = self.discblock3(output)  # 第三个残差块组

        # 将输出特征展平为 (N, disc_ch*16*4*4)
        output = output.view(-1, self.disc_ch * 16 * 4 * 4)
        # 条件部分：将条件嵌入 y 经过 linear2 得到和输出特征相同维度的向量，计算内积作为条件得分
        output_y = torch.sum(output * self.linear2(y), 1, keepdim=True)
        # 无条件部分得分与条件得分相加
        output = self.linear1(output) + output_y

        return output.view(-1, 1)


######################################################################################################################
# 主程序：测试生成器和判别器
if __name__ == "__main__":
    # 构造生成器和判别器，并移动到指定设备（device）
    netG = sngan_generator(nz=256, dim_embed=128).to(device)
    netD = sngan_discriminator(dim_embed=128).to(device)

    # 测试批量：生成 4 个样本
    N = 4
    z = torch.randn(N, 256).to(device)  # 随机噪声向量 z
    y = torch.randn(N, 128).to(device)  # 条件嵌入向量 y（通常应由标签嵌入网络生成）
    x = netG(z, y)  # 生成器生成图像
    o = netD(x, y)  # 判别器对生成图像进行判别
    print(x.size())  # 输出生成图像尺寸
    print(o.size())  # 输出判别器得分尺寸


    # 辅助函数：计算模型参数总数和可训练参数数量
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}


    print(get_parameter_number(netG))
    print(get_parameter_number(netD))
