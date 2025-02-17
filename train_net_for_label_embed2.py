import os
import timeit

import numpy as np
import torch
import torch.nn as nn

from config.config import device


# -------------------------------------------------------------
def train_net_embed(net, net_name, trainloader, testloader, epochs=200, resume_epoch=0,
        lr_base=0.01, lr_decay_factor=0.1, lr_decay_epochs=[80, 140],
        weight_decay=1e-4, path_to_ckpt=None):
    """
    训练图像嵌入网络（x2y），即将输入图像映射到标签空间（或嵌入空间）上。

    该函数用于训练一个 CNN 模型，使其输出与真实标签（经过归一化后的标签）尽可能接近，
    损失函数采用 MSELoss，同时使用 SGD 优化器进行训练，并在指定 epoch 处降低学习率。
    若指定了 checkpoint 路径且 resume_epoch > 0，则加载对应的模型和优化器状态以实现断点续训。

    参数:
        net (torch.nn.Module): 待训练的图像嵌入网络模型（例如基于 ResNet 的模型）。
        net_name (str): 模型名称，用于标识不同的网络（一般在日志或保存时使用）。
        trainloader (DataLoader): 训练集 DataLoader，返回 (batch_train_images, batch_train_labels)。
        testloader (DataLoader or None): 测试集 DataLoader，若为 None，则只输出训练集损失。
        epochs (int, optional): 总训练轮数，默认为 200。
        resume_epoch (int, optional): 从第几轮开始恢复训练，默认为 0，即从头训练。
        lr_base (float, optional): 初始学习率，默认为 0.01。
        lr_decay_factor (float, optional): 学习率衰减因子，默认为 0.1。
        lr_decay_epochs (list of int, optional): 触发学习率衰减的 epoch 列表，例如 [80, 140]。
        weight_decay (float, optional): 权重衰减（L2 正则化）系数，默认为 1e-4。
        path_to_ckpt (str or None, optional): 模型 checkpoint 存放路径；若不为 None 且 resume_epoch > 0，则加载对应的 checkpoint。

    返回:
        torch.nn.Module: 训练后的图像嵌入网络模型（net）。
    """

    # 内部函数：调整学习率，根据当前 epoch 对学习率进行衰减
    def adjust_learning_rate_1(optimizer, epoch):
        """根据 lr_decay_epochs 列表中的设定降低学习率。"""
        lr = lr_base
        num_decays = len(lr_decay_epochs)
        for decay_i in range(num_decays):
            if epoch >= lr_decay_epochs[decay_i]:
                lr = lr * lr_decay_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    net = net.to(device)  # 将模型移动到 GPU 上
    criterion = nn.MSELoss()  # 使用均方误差损失
    optimizer = torch.optim.SGD(net.parameters(), lr=lr_base, momentum=0.9,
                                weight_decay=weight_decay)

    # 如果指定了 checkpoint 路径且 resume_epoch > 0，则加载断点续训数据
    if path_to_ckpt is not None and resume_epoch > 0:
        save_file = path_to_ckpt + "/embed_x2y_ckpt_in_train/embed_x2y_checkpoint_epoch_{}.pth".format(
            resume_epoch)
        checkpoint = torch.load(save_file)
        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])

    start_tmp = timeit.default_timer()
    # 开始训练，每个 epoch 内遍历训练集
    for epoch in range(resume_epoch, epochs):
        net.train()
        train_loss = 0
        adjust_learning_rate_1(optimizer, epoch)  # 调整当前 epoch 的学习率
        for _, (batch_train_images, batch_train_labels) in enumerate(trainloader):
            batch_train_images = batch_train_images.type(torch.float).to(device)
            batch_train_labels = batch_train_labels.type(torch.float).view(-1, 1).to(device)

            # 前向传播：得到模型输出；注意此处模型返回的是 (outputs, features)，这里只取 outputs
            outputs, _ = net(batch_train_images)
            loss = criterion(outputs, batch_train_labels)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()
        train_loss = train_loss / len(trainloader)

        # 若 testloader 不为 None，则在测试集上计算损失
        if testloader is None:
            print('Train net_x2y for embedding: [epoch %d/%d] train_loss:%f Time:%.4f' %
                  (epoch + 1, epochs, train_loss, timeit.default_timer() - start_tmp))
        else:
            net.eval()  # 设定为评估模式，BatchNorm 使用移动平均
            with torch.no_grad():
                test_loss = 0
                for batch_test_images, batch_test_labels in testloader:
                    batch_test_images = batch_test_images.type(torch.float).to(device)
                    batch_test_labels = batch_test_labels.type(torch.float).view(-1, 1).to(device)
                    outputs, _ = net(batch_test_images)
                    loss = criterion(outputs, batch_test_labels)
                    test_loss += loss.cpu().item()
                test_loss = test_loss / len(testloader)
                print(
                    'Train net_x2y for label embedding: [epoch %d/%d] train_loss:%f test_loss:%f Time:%.4f' %
                    (epoch + 1, epochs, train_loss, test_loss, timeit.default_timer() - start_tmp))

        # 保存 checkpoint：每 50 个 epoch 或最后一个 epoch 保存一次模型
        if path_to_ckpt is not None and (((epoch + 1) % 50 == 0) or (epoch + 1 == epochs)):
            save_file = path_to_ckpt + "/embed_x2y_ckpt_in_train/embed_x2y_checkpoint_epoch_{}.pth".format(
                epoch + 1)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rng_state': torch.get_rng_state()
            }, save_file)
    return net


###################################################################################
class label_dataset(torch.utils.data.Dataset):
    """
    自定义标签数据集，用于训练 label 到嵌入空间的映射网络（y2h）。

    该数据集只包含标签数据，每个样本为一个标签（标量）。
    """

    def __init__(self, labels):
        """
        参数:
            labels (np.ndarray 或 list): 标签数据数组
        """
        super(label_dataset, self).__init__()
        self.labels = labels
        self.n_samples = len(self.labels)

    def __getitem__(self, index):
        """
        根据索引返回对应的标签。

        参数:
            index (int): 样本索引

        返回:
            单个标签数据
        """
        y = self.labels[index]
        return y

    def __len__(self):
        """返回数据集中样本总数。"""
        return self.n_samples


def train_net_y2h(unique_labels_norm, net_y2h, net_embed, epochs=500, lr_base=0.01,
        lr_decay_factor=0.1, lr_decay_epochs=[150, 250, 350], weight_decay=1e-4, batch_size=128):
    """
    训练标签映射网络（y2h），即将归一化后的标签映射到与图像嵌入空间一致的高维表示上。

    训练过程的思路：
      1. 首先定义一个 label_dataset，将所有唯一归一化标签包装成数据集，
         并用 DataLoader 加载。
      2. 固定 net_embed（预训练好的图像嵌入网络）的 eval 模式，从中获取 h2y 部分，
         用于将嵌入转换回标签。
      3. 对输入标签加上高斯噪声后，通过 net_y2h 映射到嵌入空间，再通过 net_embed.h2y 得到重构标签，
         并计算重构损失（MSE）。
      4. 使用 SGD 优化器更新 net_y2h 参数，同时采用学习率衰减策略。

    参数:
        unique_labels_norm (np.ndarray): 归一化后的唯一标签数组，数值范围应在 [0,1] 内。
        net_y2h (torch.nn.Module): 待训练的标签映射网络，将标签映射到嵌入空间。
        net_embed (torch.nn.Module): 预训练好的图像嵌入网络，用于提取 h2y 部分，将嵌入映射回标签空间。
        epochs (int, optional): 训练轮数，默认为 500。
        lr_base (float, optional): 初始学习率，默认为 0.01。
        lr_decay_factor (float, optional): 学习率衰减因子，默认为 0.1。
        lr_decay_epochs (list of int, optional): 学习率衰减的轮数列表，如 [150, 250, 350]。
        weight_decay (float, optional): 权重衰减系数，默认为 1e-4。
        batch_size (int, optional): 训练时的批量大小，默认为 128。

    返回:
        torch.nn.Module: 训练后的标签映射网络（net_y2h）。
    """

    # 内部函数：调整学习率
    def adjust_learning_rate_2(optimizer, epoch):
        """根据 lr_decay_epochs 列表中的设定降低学习率。"""
        lr = lr_base
        num_decays = len(lr_decay_epochs)
        for decay_i in range(num_decays):
            if epoch >= lr_decay_epochs[decay_i]:
                lr = lr * lr_decay_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # 保证标签归一化后的数值在 [0,1] 内
    assert np.max(unique_labels_norm) <= 1 and np.min(unique_labels_norm) >= 0
    trainset = label_dataset(unique_labels_norm)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    net_embed.eval()  # 固定 net_embed，不更新参数
    # 获取 net_embed 中的 h2y 层（将嵌入映射回标签），用于辅助训练 net_y2h
    net_h2y = net_embed.h2y
    optimizer_y2h = torch.optim.SGD(net_y2h.parameters(), lr=lr_base, momentum=0.9,
                                    weight_decay=weight_decay)

    start_tmp = timeit.default_timer()
    for epoch in range(epochs):
        net_y2h.train()
        train_loss = 0
        adjust_learning_rate_2(optimizer_y2h, epoch)
        for _, batch_labels in enumerate(trainloader):
            batch_labels = batch_labels.type(torch.float).view(-1, 1).to(device)

            # 为当前批次标签生成噪声（服从 N(0, 0.2)）
            batch_size_curr = len(batch_labels)
            batch_gamma = np.random.normal(0, 0.2, batch_size_curr)
            batch_gamma = torch.from_numpy(batch_gamma).view(-1, 1).type(torch.float).to(device)

            # 将噪声添加到标签上，并用 clamp 限制在 [0,1] 内
            batch_labels_noise = torch.clamp(batch_labels + batch_gamma, 0.0, 1.0)

            # 前向传播：将加噪标签通过 net_y2h 得到嵌入，再通过 net_h2y 重构标签
            batch_hiddens_noise = net_y2h(batch_labels_noise)
            batch_rec_labels_noise = net_h2y(batch_hiddens_noise)

            # 计算重构损失（MSE），使得重构后的标签与加噪标签接近
            loss = nn.MSELoss()(batch_rec_labels_noise, batch_labels_noise)

            optimizer_y2h.zero_grad()
            loss.backward()
            optimizer_y2h.step()

            train_loss += loss.cpu().item()
        train_loss = train_loss / len(trainloader)

        print('\n Train net_y2h: [epoch %d/%d] train_loss:%f Time:%.4f' %
              (epoch + 1, epochs, train_loss, timeit.default_timer() - start_tmp))
    return net_y2h