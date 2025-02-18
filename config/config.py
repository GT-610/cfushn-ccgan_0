# -*- coding: utf-8 -*-
# @Time    : 2025/2/17 19:17
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt

from .constants import *

# --------------------------- env相关设置 ---------------------------
# os.chdir(ROOT_PATH)
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
print(f"device is {device}\n")
