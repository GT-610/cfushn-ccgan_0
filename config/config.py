# -*- coding: utf-8 -*-
# @Time    : 2025/2/17 19:17
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt

from constants import *

# --------------------------- env相关设置 ---------------------------
os.chdir(ROOT_PATH)
N_GPU = torch.cuda.device_count()
# 如果指定了 torch 模型保存路径，则设置环境变量 TORCH_HOME
if TORCH_MODEL_PATH != "None":
    os.environ['TORCH_HOME'] = TORCH_MODEL_PATH
plt.switch_backend('agg')  # 使用非交互式后端

# --------------------------- 设置随机种子，确保结果可复现 ---------------------------
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(SEED)

# --------------------------- 选定device ---------------------------
device = None
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"  # macOS m1 的 mps ≈ NVIDIA 1050Ti
else:
    device = "cpu"
print(f"device is {device}")

# --------------------------- 创建输出文件夹 ---------------------------
path_to_output = os.path.join(ROOT_PATH,
                              'output/CcGAN_{}_{}_si{:.3f}_ka{:.3f}_{}_nDs{}_nDa{}_nGa{}_Dbs{}_Gbs{}_v2'.format(
                                      GAN_ARCH, THRESHOLD_TYPE, KERNEL_SIGMA, KAPPA, LOSS_TYPE,
                                      NUM_D_STEPS, NUM_GRAD_ACC_D, NUM_GRAD_ACC_G,
                                      BATCH_SIZE_D, BATCH_SIZE_G))
os.makedirs(path_to_output, exist_ok=True)
save_models_folder = os.path.join(path_to_output, 'saved_models')
os.makedirs(save_models_folder, exist_ok=True)
save_images_folder = os.path.join(path_to_output, 'saved_images')
os.makedirs(save_images_folder, exist_ok=True)
path_to_embed_models = os.path.join(ROOT_PATH, 'output/embed_models')
os.makedirs(path_to_embed_models, exist_ok=True)
