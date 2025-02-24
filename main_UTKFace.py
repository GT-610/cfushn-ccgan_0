# -*- coding: utf-8 -*-
# @Time    : 2025/2/19 17:54
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

from config import cfg
from flow import process_flow

# torch.cuda.set_device(0)  # 设置当前 GPU 设备为 0 (如不指定,默认并行)

# cfg.version = "v022117"  # 固定版本
cfg.dataset_name = "UTKFace"
cfg.class_label_h5_key = "races"
cfg.img_size = 64
cfg.num_classes = 5
cfg.min_label = 1  # scoping 1~60岁,用于数据筛选与归一化
cfg.max_label = 60  # scoping 1~60岁,用于数据筛选与归一化
cfg.min_img_num_per_label = 60
cfg.max_img_num_per_label = 9999
cfg.threshold_type = "soft"
cfg.loss_type = "vanilla"

# 训练epoch或iter调成1,用于测试
cfg.n_iters = 40000
cfg.epoch_cnn_embed = 200
cfg.epoch_net_y2h = 500

# 评估相关
cfg.comp_fid = True
cfg.dump_fake_for_niqe = False  # 是否导出用于 NIQE 计算的图像
cfg.comp_is_and_fid_only = False  # 是否只计算 IS 和 FID（减少计算量）
cfg.n_fake_per_label = 1000  # 每个连续标签(整数)生成多少张图像用于评估
cfg.pretrained_ae_pth = "ckpt_AE_epoch_200_seed_2020_CVMode_False.pth"
cfg.pretrained_cnn4cont_pth = "ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth"
cfg.pretrained_cnn4class_pth = "ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_200_seed_2020_classify_5_races_CVMode_False.pth"

# 算法主体流程
process_flow()
