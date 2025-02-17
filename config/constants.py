# -*- coding: utf-8 -*-
# @Time    : 2025/2/13 14:33
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

root_path = "/home/cy/workdir/cfushn-ccgan_0"
data_path = "/home/cy/workdir/cfushn-ccgan_0/data"
eval_path = "/home/cy/workdir/cfushn-ccgan_0/evaluation"
niqe_dump_path = "/home/cy/workdir/cfushn-ccgan_0/NIQE_64x64/fake_data"
torch_model_path = 'None'

seed = 2021  # 随机数种子
num_workers = 0  # 数据加载时使用的线程数

niters = 40000  # GAN 训练的总迭代次数
resume_niters = 0
# resume_niter=40000
## 设置为0,则从头开始训练；设置为其他值，则载入相应的checkpoint（若有），然后继续训练。
save_niters_freq = 2000  # 模型 checkpoint 的保存频率（以迭代次数计）
visualize_freq = 500  # 每隔多少次迭代进行一次生成图像的可视化

min_label = 1
max_label = 60  # 标签的最大值（用于回归标签的归一化或裁剪）
num_classes = 5
img_size = 64
num_channels = 3  # 图像的通道数（例如 RGB 图像为 3）
max_num_img_per_label = 99999  # 当前在所有类别里能找到的最大类别样本数
max_num_img_per_label_after_replica = 200  # 预先想要为每个类别（标签）在过采样之后所达到的最大样本数。

base_lr_x2y = 0.01  # 用于训练 net_embed（图像到嵌入）的基础学习率
base_lr_y2h = 0.01  # 用于训练 net_y2h（标签到嵌入）的基础学习率

num_d_steps = 2  # 每次训练中，判别器更新的步数
kernel_sigma = -1.0
kappa = -1.0
lr_g = 1e-4
lr_d = 1e-4
batch_size_d = 256  # 判别器训练时的批量大小
batch_size_g = 256  # 生成器训练时的批量大小
num_grad_acc_d = 1  # 判别器梯度累积的步数（多个小批次累积后再更新）
num_grad_acc_g = 1

GAN_arch = "SNGAN"
loss_type = "vanilla"
threshold_type = "hard"  # 邻域阈值类型：'hard'（硬阈值）或 'soft'（软阈值），用于选择真实样本的邻域
nonzero_soft_weight_threshold = 1e-3  # 软阈值下用于确定非零权重的阈值（用于 SVDL 损失计算）
net_embed_type = "ResNet34_embed"
epoch_cnn_embed = 200
resumeepoch_cnn_embed = 0
epoch_net_y2h = 500
batch_size_embed = 256

dim_gan = 256  # 生成器输入的噪声向量的维度
dim_embed = 128

use_DiffAugment = False  # 是否启用 DiffAugment 数据增强技术的标志
policy = 'color,translation,cutout'  # DiffAugment 的具体策略（定义了使用哪些数据增强操作）

