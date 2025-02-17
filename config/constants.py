# -*- coding: utf-8 -*-
# @Time    : 2025/2/13 14:33
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

ROOT_PATH = "/home/cy/workdir/cfushn-ccgan_0"
DATA_PATH = "/home/cy/workdir/cfushn-ccgan_0/data"
EVAL_PATH = "/home/cy/workdir/cfushn-ccgan_0/evaluation"
NIQE_DUMP_PATH = "/home/cy/workdir/cfushn-ccgan_0/NIQE_64x64/fake_data"
TORCH_MODEL_PATH = 'None'

SEED = 2021  # 随机数种子
NUM_WORKERS = 0  # 数据加载时使用的线程数

N_ITERS = 40000  # GAN 训练的总迭代次数
RESUME_N_ITERS = 0
# resume_niter=40000
## 设置为0,则从头开始训练；设置为其他值，则载入相应的checkpoint（若有），然后继续训练。
SAVE_N_ITERS_FREQ = 2000  # 模型 checkpoint 的保存频率（以迭代次数计）
VISUALIZE_FREQ = 500  # 每隔多少次迭代进行一次生成图像的可视化

MIN_LABEL = 1
MAX_LABEL = 60  # 标签的最大值（用于回归标签的归一化或裁剪）
NUM_CLASSES = 5
IMG_SIZE = 64
DIM_GAN = 256  # 生成器输入的噪声向量的维度
DIM_EMBED = 128  # 嵌入空间的维度
MAX_NUM_IMG_PER_LABEL = 99999  # 当前在所有类别里能找到的最大类别样本数
MAX_NUM_IMG_PER_LABEL_AFTER_REPLICA = 200  # 预先想要为每个类别（标签）在过采样之后所达到的最大样本数。

BASE_LR_X2Y = 0.01  # 用于训练 net_embed（图像到嵌入）的基础学习率
BASE_LR_Y2H = 0.01  # 用于训练 net_y2h（标签到嵌入）的基础学习率

NUM_D_STEPS = 2  # 每次训练中，判别器更新的步数
KERNEL_SIGMA = -1.0
KAPPA = -1.0
LR_G = 1e-4
LR_D = 1e-4
BATCH_SIZE_D = 256  # 判别器训练时的批量大小
BATCH_SIZE_G = 256  # 生成器训练时的批量大小
NUM_GRAD_ACC_D = 1  # 判别器梯度累积的步数（多个小批次累积后再更新）
NUM_GRAD_ACC_G = 1  # 生成器梯度累积的步数（多个小批次累积后再更新）

GAN_ARCH = "SNGAN"
LOSS_TYPE = "vanilla"
THRESHOLD_TYPE = "hard"  # 邻域阈值类型：'hard'（硬阈值）或 'soft'（软阈值），用于选择真实样本的邻域
NONZERO_SOFT_WEIGHT_THRESHOLD = 1e-3  # 软阈值下用于确定非零权重的阈值（用于 SVDL 损失计算）
NET_EMBED_TYPE = "ResNet34_embed"
EPOCH_CNN_EMBED = 200
RESUME_EPOCH_CNN_EMBED = 0
EPOCH_NET_Y2H = 500
BATCH_SIZE_EMBED = 256

USE_DiffAugment = False  # 是否启用 DiffAugment 数据增强技术的标志
POLICY = 'color,translation,cutout'  # DiffAugment 的具体策略（定义了使用哪些数据增强操作）
