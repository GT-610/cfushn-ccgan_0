# -*- coding: utf-8 -*-
# @Time    : 2025/2/19 17:54
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

from datetime import datetime

from config.config import *
from data_process import data_process
from eval import evaluate
from train import train_process

"""
kill -10 PID: 在训练过程中,立即采样一次
kill -12 PID: 立即保存当前ckpt,并退出进程
"""

# ------------------------------- 数据处理 -------------------------------
origin_data, data, kernel_sigma, kappa = data_process()

# ------------------------------- 确定目录 -------------------------------
version_ = datetime.now().strftime("%m%d%H")
# version_ = "021920" # 指定版本
path_to_output = os.path.join(
        ROOT_PATH, f'output/'
                   f'CcGAN_{GAN_ARCH}_{THRESHOLD_TYPE}'
                   f'_si{kernel_sigma:.3f}_ka{kappa:.3f}'
                   f'_{LOSS_TYPE}_nDs{NUM_D_STEPS}_nDa{NUM_GRAD_ACC_D}_nGa{NUM_GRAD_ACC_G}'
                   f'_Dbs{BATCH_SIZE_D}_Gbs{BATCH_SIZE_G}_v{version_}')
os.makedirs(path_to_output, exist_ok=True)
print(path_to_output, '\n')

# ------------------------------- 模型训练 -------------------------------
netG, net_y2h = train_process(data, kernel_sigma, kappa, path_to_output)

# ------------------------------- 模型评估 -------------------------------
evaluate(origin_data, netG, net_y2h, path_to_output)
