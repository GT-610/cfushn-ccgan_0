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
os.chdir(root_path)
NGPU = torch.cuda.device_count()
# 如果指定了 torch 模型保存路径，则设置环境变量 TORCH_HOME
if torch_model_path != "None":
    os.environ['TORCH_HOME'] = torch_model_path
plt.switch_backend('agg')  # 使用非交互式后端

# --------------------------- 设置随机种子，确保结果可复现 ---------------------------
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(seed)

# --------------------------- 选定device ---------------------------
device = ""
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"  # macOS m1 的 mps ≈ NVIDIA 1050Ti
else:
    device = "cpu"
print(f"device is {device}")

# --------------------------- 创建输出文件夹 ---------------------------
path_to_output = os.path.join(root_path,
                              'output/CcGAN_{}_{}_si{:.3f}_ka{:.3f}_{}_nDs{}_nDa{}_nGa{}_Dbs{}_Gbs{}_v2'.format(
                                      GAN_arch, threshold_type, kernel_sigma, kappa, loss_type,
                                      num_d_steps, num_grad_acc_d, num_grad_acc_g,
                                      batch_size_d, batch_size_g))
os.makedirs(path_to_output, exist_ok=True)
save_models_folder = os.path.join(path_to_output, 'saved_models')
os.makedirs(save_models_folder, exist_ok=True)
save_images_folder = os.path.join(path_to_output, 'saved_images')
os.makedirs(save_images_folder, exist_ok=True)
path_to_embed_models = os.path.join(root_path, 'output/embed_models')
os.makedirs(path_to_embed_models, exist_ok=True)
