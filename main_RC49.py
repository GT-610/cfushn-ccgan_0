# -*- coding: utf-8 -*-
# @Time    : 2025/2/19 17:54
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

from config import cfg
from flow import process_flow

# torch.cuda.set_device(0)  # 设置当前 GPU 设备为 0 (如不指定,默认并行)

# cfg.version = "v022117"  # 固定版本
cfg.dataset_name = "RC-49_filtered"  # 原数据集图片量过大,筛选后另置一个数据集
cfg.class_label_h5_key = "types"
cfg.img_size = 64
# 这个49是固定的,是特定于[0,90]数据集的总类别!
# 但是要注意,数据集中存储的0~49,其中缺6,这些应视作类别标签, 而非类别标签索引! 计算交叉熵时候必须用索引!
# 为了简化实验,仅在数据处理时打印一下标签与索引对应关系,后续的class_labels存储的都是索引!
cfg.num_classes = 49
cfg.min_label = 0.0  # scoping 0~90度的数据,用于数据筛选与归一化
cfg.max_label = 90.0  # scoping 0~90度的数据,用于数据筛选与归一化
cfg.min_img_num_per_label = 5
cfg.max_img_num_per_label = 9999
cfg.threshold_type = "soft"
cfg.loss_type = "hinge"
cfg.kappa = -2.0

# 评估相关
cfg.if_eval = True
cfg.dump_fake_for_niqe = False  # 是否导出用于 NIQE 计算的图像
cfg.comp_is_and_fid_only = False  # 是否只计算 IS 和 FID（减少计算量）
cfg.n_fake_per_label = 200  # 每个连续标签(整数)生成多少张图像用于评估
cfg.pretrained_ae_pth = "ckpt_AE_epoch_200_seed_2020_CVMode_False.pth"
cfg.pretrained_cnn4cont_pth = "ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth"
cfg.pretrained_cnn4class_pth = "ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_200_seed_2020_classify_49_chair_types_CVMode_False.pth"

# 算法流程
process_flow()
