# # -*- coding: utf-8 -*-
# # @Time    : 2025/2/12 13:48
# # @Author  : cfushn
# # @Comments:
# # @Software: PyCharm
#
# import os
# import re
#
# import torch
#
# from utils.log_util import cy_log
#
#
# def save_checkpoint(model, optimizer, epoch, loss):
#     state = {
#         'state_dict': model.state_dict(),
#         'optimizer': optimizer.state_dict(),
#         'epoch': epoch,
#         'loss': loss
#     }
#     folder = '/home/cy/workdir/cfushn_ldm/model_save'
#     cy_log(f'save check point (epoch:{epoch})...')
#     torch.save(state, os.path.join(folder, f"checkpoint_{epoch}.pth"))
#     cy_log('save successfully.\n')
#
#
# def load_checkpoint(unet, optimizer=None, checkpoint_path: str = None):
#     """
#     读取模型的 checkpoint 文件并加载权重、优化器状态及其他训练信息。
#
#     Args:
#         unet (torch.nn.Module): 需要加载权重的模型实例。
#         optimizer (torch.optim.Optimizer, optional): 如果需要恢复优化器状态，请传入优化器实例。
#         checkpoint_path (str): 默认None,会加载特定目录中最后一个checkpoint
#
#     Returns:
#         dict: 包含额外信息的字典，例如 epoch 和损失。
#     """
#     if unet is None:
#         raise Exception("请初始化模型实例!")
#
#     if checkpoint_path is None:
#         folder = '/home/cy/workdir/cfushn_ldm/model_save'
#         files = os.listdir(folder)
#         matched_files = []
#         for file in files:
#             match = re.match("checkpoint_(\d+).pth", file)
#             if match:
#                 # 提取文件名中的 epoch 值并保存
#                 epoch = int(match.group(1))  # 假设第一个捕获组是数字
#                 matched_files.append((epoch, file))
#         # 如果没有匹配文件
#         if not matched_files:
#             raise FileNotFoundError(f"there has no matched models in '{folder}'")
#         # 按 epoch 值排序并返回最后一个文件的完整路径
#         matched_files.sort(key=lambda x: x[0])  # 按 epoch 升序排序
#         latest_file = matched_files[-1][1]  # 获取最后一个文件名
#         checkpoint_path = os.path.join(folder, latest_file)
#
#     # 加载 checkpoint 文件
#     check_point = torch.load(checkpoint_path)
#
#     unet.load_state_dict(check_point["state_dict"])
#     unet.train()
#
#     # 如果提供了优化器实例，加载优化器状态
#     optimizer is None or optimizer.load_state_dict(check_point["optimizer"])
#
#     last_epoch = check_point["epoch"]
#     loss = check_point["loss"]
#     # 返回其他训练信息
#     print(f"-------------load checkpoint: epoch:{last_epoch},loss:{loss}---------------")
#     return unet, optimizer, last_epoch, loss
