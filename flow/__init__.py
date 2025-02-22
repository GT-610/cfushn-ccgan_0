# -*- coding: utf-8 -*-
# @Time    : 2025/2/20 16:42
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

from config import init_config
from .data_process import data_process
from .eval import evaluate_process
from .train import train_process


def process_flow():
    """
    算法整体流程

    数据集中存储的是类别标签, 而非类别标签索引! 计算交叉熵时候必须用索引!
    为了简化实验,仅在数据处理时打印一下对应关系,后续的class_labels存储的都是索引!
    """

    # ------------------------------- 一些初始化 -------------------------------
    init_config()

    # ------------------------------- 数据处理 -------------------------------
    origin_data, data = data_process()

    # ------------------------------- 模型训练 -------------------------------
    netG, net_y2h = train_process(data)

    # ------------------------------- 模型评估 -------------------------------
    evaluate_process(origin_data, netG, net_y2h)
