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
GPU_PARALLEL = True  # 是否使用多GPU并行训练(如果指定CUDA_VISIBLE_DEVICES为单个,则该选项无效)
NUM_WORKERS = 0  # 数据加载时使用的线程数

N_ITERS = 40000  # GAN 训练的总迭代次数
RESUME_N_ITERS = 0  # 恢复训练,0-从头训练,非0-从指定迭代次数开始训练(若找不到指定,则寻找最新ckpt)
SAVE_N_ITERS_FREQ = 2000  # 模型 checkpoint 的保存频率（以迭代次数计）
VISUALIZE_FREQ = 500  # 每隔多少次迭代进行一次生成图像的可视化

MIN_IMG_NUM_PER_LABEL = 60  # 每个标签(组合)对应的最小样本数(达不到就随机复制)
MAX_IMG_NUM_PER_LABEL = 300  # 每个标签(组合)对应的最大样本数(超过了就随机删除)
MIN_LABEL = 1
MAX_LABEL = 60  # 标签的最大值（用于回归标签的归一化或裁剪）
NUM_CLASSES = 5
IMG_SIZE = 64
DIM_GAN = 256  # 生成器输入的噪声向量的维度
DIM_EMBED = 128  # 嵌入空间的维度

BASE_LR_X2Y = 0.01  # 用于训练 net_embed（图像到嵌入）的基础学习率
BASE_LR_Y2H = 0.01  # 用于训练 net_y2h（标签到嵌入）的基础学习率

NUM_D_STEPS = 2  # 每次训练中，判别器更新的步数
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

USE_DiffAugment = True  # 是否启用 DiffAugment 数据增强技术的标志
# POLICY = 'color,translation,cutout'  # DiffAugment 的具体策略（定义了使用哪些数据增强操作）
POLICY = 'translation,cutout'  # DiffAugment 的具体策略（定义了使用哪些数据增强操作）

''' Sampling and Evaluation '''
SAMP_BATCH_SIZE = 200
N_FAKE_PER_LABEL = 1000
COMP_FID = False
EPOCH_FID_CNN = 200
FID_RADIUS = 0
DUMP_FAKE_FOR_NIQE = False
COMP_IS_AND_FID_ONLY = False
