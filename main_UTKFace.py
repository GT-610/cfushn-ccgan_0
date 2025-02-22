# -*- coding: utf-8 -*-
# @Time    : 2025/2/19 17:54
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

import torch

from config import cfg
from flow import process_flow

# torch.cuda.set_device(0)  # 设置当前 GPU 设备为 0

# cfg.version = "v022118"
cfg.dataset_name = "UTKFace"
cfg.cont_label_h5_key = "labels"
cfg.class_label_h5_key = "races"
cfg.img_size = 64
cfg.num_classes = 5
cfg.min_label = 1  # 只看1~60岁,用于数据筛选与归一化
cfg.max_label = 60  # 只看1~60岁,用于数据筛选与归一化
cfg.min_img_num_per_label = 60
cfg.max_img_num_per_label = 9999
cfg.threshold_type = "soft"

# 训练epoch或iter调成1,用于测试
cfg.n_iters = 40000
cfg.epoch_cnn_embed = 200
cfg.epoch_net_y2h = 500

# 算法主体流程
process_flow()
