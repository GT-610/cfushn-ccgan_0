# -*- coding: utf-8 -*-
# @Time    : 2025/2/19 17:54
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

from config import cfg
from flow import process_flow

# cfg.version = "v022117"
cfg.dataset_name = "RC-49_filtered"
cfg.cont_label_h5_key = "labels"
cfg.class_label_h5_key = "types"
cfg.img_size = 64
# 这个49是固定的,是特定于[min_label,max_label]数据集的总类别!
# 但是要注意,数据集中存储的0~49,其中缺6,这些是类别标签, 而非类别标签索引! 计算交叉熵时候必须用索引!
# 为了简化实验,仅在数据处理时打印一下对应关系,后续的class_labels存储的都是索引!
cfg.num_classes = 49
cfg.min_label = 0  # 只看0~90度的数据,用于数据筛选与归一化
cfg.max_label = 90  # 只看0~90度的数据,用于数据筛选与归一化
cfg.min_img_num_per_label = 5
cfg.max_img_num_per_label = 9999
cfg.threshold_type = "soft"
cfg.loss_type = "hinge"

cfg.kappa = -2.0

# 算法流程
process_flow()
