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
    # criterion = nn.MSELoss()  # 使用均方误差损失
    # 1. 连续标签的回归损失，使用均方误差
    criterion_cont = nn.MSELoss()
    # 2. 离散标签的分类损失，使用交叉熵（注意：交叉熵损失要求目标为 long 类型，并且不需要 one-hot 编码）
    criterion_class = nn.CrossEntropyLoss()

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
        for _, (batch_train_images, batch_train_labels_cont, batch_train_labels_class) in enumerate(
                trainloader):
            batch_train_images = batch_train_images.type(torch.float).to(device)
            # batch_train_labels = batch_train_labels.type(torch.float).view(-1, 1).to(device)
            batch_train_labels_cont = batch_train_labels_cont.type(torch.float).view(-1, 1).to(
                    device)
            # 离散标签需要为 long 类型，且不 reshape
            batch_train_labels_class = batch_train_labels_class.type(torch.long).to(device)

            # 前向传播：得到模型输出；模型返回 (y_cont, y_class, features)，这里只取前二者
            y_cont, y_class, _ = net(batch_train_images)

            # 计算回归损失：连续标签与预测连续标签之间的 MSE
            loss_cont = criterion_cont(y_cont, batch_train_labels_cont)
            # 计算分类损失：离散标签预测 (logits) 与真实标签之间的交叉熵损失
            loss_class = criterion_class(y_class, batch_train_labels_class)
            loss = loss_cont + loss_class  # 综合,后续看情况可以设置权重进行调节

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
                for batch_test_images, batch_test_labels_cont, batch_test_labels_class in testloader:
                    batch_test_images = batch_test_images.type(torch.float).to(device)
                    batch_test_labels_cont = batch_test_labels_cont.type(
                            torch.float).view(-1, 1).to(device)
                    batch_test_labels_class = batch_test_labels_class.type(torch.long).to(device)
                    y_cont, y_class, _ = net(batch_test_images)
                    loss_cont = criterion_cont(y_cont, batch_test_labels_cont)
                    loss_class = criterion_class(y_class, batch_test_labels_class)
                    loss = loss_cont + loss_class
                    test_loss += loss.cpu().item()
                test_loss = test_loss / len(testloader)
                print(
                        'Train net_x2y for label embedding: [epoch %d/%d] train_loss:%f test_loss:%f Time:%.4f' %
                        (epoch + 1, epochs, train_loss, test_loss,
                         timeit.default_timer() - start_tmp))

        # 保存 checkpoint：每 50 个 epoch 或最后一个 epoch 保存一次模型
        if path_to_ckpt is not None and (((epoch + 1) % 50 == 0) or (epoch + 1 == epochs)):
            save_file = path_to_ckpt + "/embed_x2y_ckpt_in_train/embed_x2y_checkpoint_epoch_{}_v2.pth".format(
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
# 数据集类：返回连续标签和离散标签
###################################################################################
class label_dataset(torch.utils.data.Dataset):
    """
    自定义标签数据集，用于训练标签映射网络（net_y2h）。

    该数据集包含两个标签：
      - 连续标签：例如年龄（归一化后的值，float，形状为 (batch_size, 1)）。
      - 离散标签：例如种族类别（类别索引，long，取值范围 [0, num_classes-1]）。
    """

    def __init__(self, cont_labels, class_labels):
        super(label_dataset, self).__init__()
        self.cont_labels = np.array(cont_labels).astype(np.float32)
        self.class_labels = np.array(class_labels).astype(np.int64)
        assert len(self.cont_labels) == len(self.class_labels)
        self.n_samples = len(self.cont_labels)

    def __getitem__(self, index):
        y_cont = self.cont_labels[index]
        y_class = self.class_labels[index]
        return y_cont, y_class

    def __len__(self):
        """返回数据集中样本总数。"""
        return self.n_samples


###################################################################################
# 训练标签映射网络（net_y2h）的新版本
###################################################################################
def train_net_y2h(cont_labels, class_labels, net_y2h, net_embed, epochs=500, lr_base=0.01,
        lr_decay_factor=0.1, lr_decay_epochs=[150, 250, 350], weight_decay=1e-4, batch_size=128):
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
    assert np.max(cont_labels) <= 1 and np.min(cont_labels) >= 0

    # 构造标签数据集，返回 (连续标签, 离散标签)
    trainset = label_dataset(cont_labels, class_labels)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    net_embed.eval()  # 固定 net_embed，不更新参数
    # 从 net_embed 中获取两个分支：用于重构连续标签和离散标签
    net_h2y_cont = net_embed.h2y_cont
    net_h2y_class = net_embed.h2y_class
    optimizer_y2h = torch.optim.SGD(net_y2h.parameters(), lr=lr_base, momentum=0.9,
                                    weight_decay=weight_decay)

    # 定义损失函数
    criterion_cont = nn.MSELoss()  # 连续标签回归损失
    criterion_class = nn.CrossEntropyLoss()  # 离散标签分类损失

    net_y2h = net_y2h.to(device)
    start_tmp = timeit.default_timer()
    for epoch in range(epochs):
        net_y2h.train()
        train_loss = 0
        adjust_learning_rate_2(optimizer_y2h, epoch)
        for _, (batch_labels_cont, batch_labels_class) in enumerate(trainloader):
            # 将连续标签转换为 float 型，形状 (batch_size, 1)
            batch_labels_cont = batch_labels_cont.type(torch.float).view(-1, 1).to(device)
            # 将离散标签转换为 long 型，形状 (batch_size,)
            batch_labels_class = batch_labels_class.type(torch.long).view(-1).to(device)

            # 为连续标签添加噪声，噪声服从 N(0, 0.2)
            batch_size_curr = batch_labels_cont.size(0)
            batch_gamma = np.random.normal(0, 0.2, batch_size_curr)
            batch_gamma = torch.from_numpy(batch_gamma).view(-1, 1).type(torch.float).to(device)
            # 加噪后的连续标签，并 clamp 到 [0,1]
            batch_labels_cont_noise = torch.clamp(batch_labels_cont + batch_gamma, 0.0, 1.0)

            # 前向传播：将成对的标签输入 net_y2h 得到嵌入 h
            h = net_y2h(batch_labels_cont_noise, batch_labels_class)
            # 利用预训练好的 net_embed 分支重构连续标签和离散标签
            rec_cont = net_h2y_cont(h)  # 预测连续标签，形状 (batch_size, 1)
            rec_class = net_h2y_class(h)  # 预测离散标签 logits，形状 (batch_size, NUM_CLASSES)

            # 计算损失：
            # 连续标签：均方误差
            loss_cont = criterion_cont(rec_cont, batch_labels_cont_noise)
            # 离散标签：交叉熵损失；注意：CrossEntropyLoss 要求预测 logits 和目标类别索引
            loss_class = criterion_class(rec_class, batch_labels_class)
            # 总损失为两部分之和（也可以加权融合）
            loss = loss_cont + loss_class

            optimizer_y2h.zero_grad()
            loss.backward()
            optimizer_y2h.step()

            train_loss += loss.cpu().item()
        train_loss = train_loss / len(trainloader)

        print('\n Train net_y2h: [epoch %d/%d] train_loss:%f Time:%.4f' %
              (epoch + 1, epochs, train_loss, timeit.default_timer() - start_tmp))
    return net_y2h
