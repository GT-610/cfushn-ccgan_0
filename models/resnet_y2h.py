# -*- coding: utf-8 -*-
# @Time    : 2025/2/20 15:24
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm
import torch
import torch.nn as nn

from config import cfg


class ResNetY2H(nn.Module):
    """
    标签联合嵌入模型

    该网络将连续标签（例如年龄）和离散标签（例如人种）映射到与图像嵌入相同的特征空间中，
    为条件生成或其他任务提供联合标签的高维表示。

    输入:
        y_cont (torch.Tensor): 连续标签，形状应为 (batch_size, 1)。
        y_class (torch.Tensor): 离散标签，形状应为 (batch_size,) 或 (batch_size, 1)（类别索引）。

    输出:
        torch.Tensor: 融合后的标签嵌入，形状为 (batch_size, dim_embed)。
    """

    def __init__(self, dim_embed=cfg.dim_embed):
        # dim_embed (int, optional): 嵌入空间的维度，默认 DIM_EMBED (128)
        super(ResNetY2H, self).__init__()

        # 连续标签分支：将cont_dim维连续标签映射到 dim_embed 维
        self.cont_branch = nn.Sequential(
                nn.Linear(cfg.cont_dim, dim_embed),
                # 使用 GroupNorm（这里要求 dim_embed 能被分组数整除，否则可用 LayerNorm）
                nn.GroupNorm(8, dim_embed),
                nn.ReLU(),
                nn.Linear(dim_embed, dim_embed),
                nn.GroupNorm(8, dim_embed),
                nn.ReLU()
        )

        # 离散标签分支：使用嵌入层将类别索引映射到 dim_embed 维
        self.class_embed = nn.Embedding(cfg.num_classes, dim_embed)

        # 融合层：将连续和离散分支的特征拼接后，再映射到最终的嵌入空间
        self.fusion = nn.Sequential(
                nn.Linear(dim_embed * 2, dim_embed),
                nn.GroupNorm(8, dim_embed),
                nn.ReLU(),
                nn.Linear(dim_embed, dim_embed),
                nn.GroupNorm(8, dim_embed),
                nn.ReLU()
        )

    def forward(self, y_cont, y_class):
        # 确保连续标签的形状为 (batch_size, 1)，并加上一个极小值避免数值问题
        y_cont = y_cont.view(-1, cfg.cont_dim) + 1e-8
        # 经过连续分支映射
        cont_feat = self.cont_branch(y_cont)

        # 确保离散标签为长整型（类别索引），若输入形状为 (batch_size, 1)，则展平
        y_class = y_class.view(-1).long()
        # 经过嵌入层得到离散标签特征
        class_feat = self.class_embed(y_class)  # 形状为 (batch_size, dim_embed)

        # 融合：将连续和离散特征在特征维度上拼接
        combined = torch.cat([cont_feat, class_feat], dim=1)  # 形状 (batch_size, dim_embed*2)
        # 经过融合层映射到最终嵌入空间
        out = self.fusion(combined)
        return out
