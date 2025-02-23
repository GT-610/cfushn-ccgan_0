# -*- coding: utf-8 -*-
# @Time    : 2025/2/12 13:16
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm
import os
import re
import timeit
import warnings

import numpy as np
import torch
from torchvision.utils import save_image

from config import cfg
from utils.DiffAugment_pytorch import DiffAugment
from utils.img_util import img_with_sep
from utils.ipc_util import get_s2, get_s1, switch_s1, switch_s2
from utils.utils import normalize_images, hflip_images

device = cfg.device
np_rng = np.random.default_rng(cfg.seed)


def train_ccgan(kernel_sigma, kappa, images, cont_labels, class_labels, netG, netD, net_y2h,
        images_in_train_folder, ckpts_in_train_folder, clip_label=False):
    # 将网络移动到 GPU，并设置 net_y2h 为评估模式（用于标签映射）
    netG = netG.to(device)
    netD = netD.to(device)
    net_y2h = net_y2h.to(device)
    net_y2h.eval()

    optimizerG = torch.optim.Adam(netG.parameters(), lr=cfg.lr_g, betas=(0.5, 0.999),
                                  weight_decay=0)
    optimizerD = torch.optim.Adam(netD.parameters(), lr=cfg.lr_d, betas=(0.5, 0.999),
                                  weight_decay=0)

    # --------------------------- 采样准备 ---------------------------
    # 获取所有唯一的训练标签，用于随机采样
    unique_cont_labels = np.sort(np.array(list(set(cont_labels))))
    unique_class_labels = np.arange(cfg.num_classes)

    nrow = cfg.nrow  # 采样图片每行放多少个格子
    sample_num = nrow * cfg.num_classes  # 每行一个class
    z_fixed = torch.randn(sample_num, cfg.dim_gan, dtype=torch.float).to(device)
    # 根据训练集中的年龄范围（continuous label）选取 10 个等距点 (分位数, eg:q=0.5就是中位数)
    start_label = np.quantile(cont_labels, 0.05)
    end_label = np.quantile(cont_labels, 0.95)
    # 在线性空间中，生成某区间等距分布的num个数
    selected_cont_labels = np.linspace(start_label, end_label, num=nrow)
    selected_class_labels = unique_class_labels  # class类别数不多, 直接全用了
    y_cont_fixed = np.zeros(sample_num, dtype=np.float32)
    y_class_fixed = np.zeros(sample_num)
    for i in range(cfg.num_classes):
        curr_class_index = selected_class_labels[i]
        for j in range(nrow):
            # 行优先下，网格中第 i 行、第 j 列的整体索引
            idx = i * nrow + j
            y_cont_fixed[idx] = selected_cont_labels[j]
            y_class_fixed[idx] = selected_class_labels[curr_class_index]
    print(f"selected cont labels for sample (covering all classes):\n{selected_cont_labels}")
    y_cont_fixed = torch.from_numpy(y_cont_fixed).type(torch.float).view(-1, 1).to(device)
    y_class_fixed = torch.from_numpy(y_class_fixed).type(torch.long).view(-1).to(device)

    # --------------------------- 恢复训练 ---------------------------
    start_iter = 0
    if cfg.resume_n_iters > 0:
        checkpoint_path = os.path.join(ckpts_in_train_folder,
                                       f"ckpt_niter_{cfg.resume_n_iters}.pth")
        if not os.path.isfile(checkpoint_path):
            print(f"Warning: {checkpoint_path} not exists.The latest ckpt will be loaded.")
            files = os.listdir(ckpts_in_train_folder)
            matched_files = []
            for file in files:
                match = re.match("ckpt_niter_(\d+).pth", file)
                if match:
                    # 提取文件名中的 epoch 值并保存
                    num = int(match.group(1))  # 假设第一个捕获组是数字
                    matched_files.append((num, file))
            # 如果没有匹配文件
            if matched_files:
                # 按 epoch 值排序并返回最后一个文件的完整路径
                matched_files.sort(key=lambda x: x[0])  # 按 epoch 升序排序
                latest_file = matched_files[-1][1]  # 获取最后一个文件名
                start_iter = matched_files[-1][0]  # 最后一个文件名对应的迭代数
                checkpoint_path = os.path.join(ckpts_in_train_folder, latest_file)
            else:
                warnings.warn(
                        f"{ckpts_in_train_folder} has no optional ckpt. GAN will train from 0")
                # raise FileNotFoundError(f"there has no matched models in '{ckpts_in_train_folder}'")
                checkpoint_path = None
        else:
            start_iter = cfg.resume_n_iters

        # 读取ckpt
        if checkpoint_path is not None:
            print(f"Loading ckpt {start_iter} >>>")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            netG.load_state_dict(checkpoint['netG_state_dict'])
            netD.load_state_dict(checkpoint['netD_state_dict'])
            optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
            optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
            torch.set_rng_state(checkpoint['rng_state'].cpu())
            print("Loaded successfully.\n")

    # --------------------------- GAN训练 ---------------------------
    g_loss = None
    d_loss = None
    real_dis_out = None
    fake_dis_out = None
    start_time = timeit.default_timer()

    for niter in range(start_iter, cfg.n_iters):

        # --------------------------- 判别器训练部分 ---------------------------
        for _ in range(cfg.num_d_steps):
            optimizerD.zero_grad()

            # 梯度累积（多步更新平均梯度）
            for _ in range(cfg.num_grad_acc_d):
                # 从连续标签(唯一)中随机采样 batch_size 个标签
                batch_cont_labels = np_rng.choice(unique_cont_labels, size=cfg.batch_size_d,
                                                  replace=True)
                # 类别标签不再进行抽取,因为后续选出真实样本后,三者(img,cont,class)需要对应上
                # 具体流程:先对当前连续标签加噪再进行邻域操作,然后在邻域内随机选一个真实样本
                # (假设不同类别的样本在各个连续标签都很充足, 在每次的邻域择取中出现的概率差不多)
                # batch_class_labels = np.zeros(batch_size_d)

                # 对连续标签加噪：对每个标签加上 Gaussian 噪声，模拟公式中的 y_target + ε
                batch_epsilons = np_rng.normal(0, kernel_sigma, cfg.batch_size_d)
                batch_cont_labels_e = batch_cont_labels + batch_epsilons

                # 初始化数组：存放选中的真实样本索引,以及随机取到的假标签(连续)
                batch_real_index = np.zeros(cfg.batch_size_d, dtype=int)
                batch_fake_cont_labels = np.zeros(cfg.batch_size_d)

                for j in range(cfg.batch_size_d):
                    # 对于每个目标标签，寻找其邻域内的真实样本，并生成假标签
                    if cfg.threshold_type == "hard":
                        # 硬邻域：选择满足 |train_label - target_label| ≤ κ 的样本
                        index_real_in_vicinity = np.nonzero(
                                np.abs(cont_labels - batch_cont_labels_e[j]) <= kappa)[0]
                    else:
                        # 软邻域：利用逆向条件 (y - target)² ≤ -log(threshold)/κ
                        index_real_in_vicinity = np.nonzero(
                                (cont_labels - batch_cont_labels_e[j]) ** 2
                                <= -np.log(cfg.nonzero_soft_weight_threshold) / kappa)[0]

                    # 如果当前目标标签在训练集中无对应邻域样本，则重新采样（保证至少1个样本）
                    while len(index_real_in_vicinity) < 1:
                        batch_epsilons_j = np_rng.normal(0, kernel_sigma, 1)
                        batch_cont_labels_e[j] = batch_cont_labels[j] + batch_epsilons_j
                        if clip_label:
                            batch_cont_labels_e = np.clip(batch_cont_labels_e, 0.0, 1.0)
                        if cfg.threshold_type == "hard":
                            index_real_in_vicinity = np.nonzero(
                                    np.abs(cont_labels - batch_cont_labels_e[j]) <= kappa)[0]
                        else:
                            index_real_in_vicinity = np.nonzero(
                                    (cont_labels - batch_cont_labels_e[j]) ** 2
                                    <= -np.log(cfg.nonzero_soft_weight_threshold) / kappa)[0]

                    # 随机从邻域内选取一个真实样本的索引（对应于利用邻域内样本联合估计条件分布）
                    batch_real_index[j] = np_rng.choice(index_real_in_vicinity, size=1)[0]

                    # 为生成器生成假标签：在目标标签邻域内均匀采样
                    if cfg.threshold_type == "hard":
                        lb = batch_cont_labels_e[j] - kappa
                        ub = batch_cont_labels_e[j] + kappa
                    else:
                        # 软邻域使用区间 [target - sqrt(-log(threshold)/κ), target + sqrt(-log(threshold)/κ)]
                        lb = batch_cont_labels_e[j] - np.sqrt(
                                -np.log(cfg.nonzero_soft_weight_threshold) / kappa)
                        ub = batch_cont_labels_e[j] + np.sqrt(
                                -np.log(cfg.nonzero_soft_weight_threshold) / kappa)
                    # 裁剪一下
                    lb = max(0.0, lb)
                    ub = min(ub, 1.0)
                    assert lb <= ub and ub >= 0 and lb <= 1
                    # 在lb~ub的均匀分布中, 生成size=1的采样结果, 取结果数组的首个
                    batch_fake_cont_labels[j] = np_rng.uniform(lb, ub, size=1)[0]

                    # end for j

                # ---------------------- 获取真实样本 -------------------------
                # 根据选中的索引，从训练集中抽取真实图像和连续标签，(类别标签没有real/fake之分)
                # 并做数据增强（水平翻转）和归一化（对应 GAN 输入要求）
                batch_real_images = torch.from_numpy(
                        normalize_images(hflip_images(images[batch_real_index])))
                batch_real_images = batch_real_images.type(torch.float).to(device)
                batch_real_labels = cont_labels[batch_real_index]
                batch_real_labels = torch.from_numpy(batch_real_labels).type(torch.float).to(device)
                batch_class_labels = class_labels[batch_real_index]
                batch_class_labels = torch.from_numpy(batch_class_labels).type(torch.long).to(
                        device)

                # ---------------------- 生成假样本 ---------------------------
                # 将生成假标签转换为 tensor，并利用标签嵌入网络 net_y2h（新型回归标签输入机制）
                batch_fake_cont_labels = torch.from_numpy(batch_fake_cont_labels).type(
                        torch.float).to(device)
                z = torch.randn(cfg.batch_size_d, cfg.dim_gan, dtype=torch.float).to(device)
                batch_fake_images = netG(z, net_y2h(batch_fake_cont_labels, batch_class_labels))

                # 将目标标签（用于判别器条件输入）转换到 GPU
                batch_cont_labels_e = torch.from_numpy(batch_cont_labels_e).type(torch.float).to(
                        device)
                # todo: 无法得到与batch_cont_labels_e对应的真实类别标签,只能使用batch_real_labels对应的?
                # 但是我们可以假设这个假标签对应img的类别=当前真img的类别, 这就是邻域假设

                # ---------------------- 计算邻域权重 -------------------------
                # 若使用软邻域，权重根据 exp(-kappa*(y - y_target)²) 计算，
                # 对应公式中软邻域权重 w(y_i, y) = exp(-v*(y_i - y)²)
                if cfg.threshold_type == "soft":
                    real_weights = torch.exp(
                            -kappa * (batch_real_labels - batch_cont_labels_e) ** 2).to(device)
                    fake_weights = torch.exp(
                            -kappa * (batch_fake_cont_labels - batch_cont_labels_e) ** 2).to(device)
                else:
                    # 硬邻域时，每个样本权重均为1（已隐式在样本筛选中体现）
                    real_weights = torch.ones(cfg.batch_size_d, dtype=torch.float).to(device)
                    fake_weights = torch.ones(cfg.batch_size_d, dtype=torch.float).to(device)
                # end if threshold type

                # ---------------------- 判别器前向传播 ----------------------
                # 利用 DiffAugment 进行数据增强（可选）
                if cfg.use_DiffAugment:
                    real_dis_out = netD(DiffAugment(batch_real_images, policy=cfg.policy),
                                        net_y2h(batch_cont_labels_e, batch_class_labels))
                    fake_dis_out = netD(DiffAugment(batch_fake_images.detach(), policy=cfg.policy),
                                        net_y2h(batch_cont_labels_e, batch_class_labels))
                else:
                    real_dis_out = netD(batch_real_images,
                                        net_y2h(batch_cont_labels_e, batch_class_labels))
                    fake_dis_out = netD(batch_fake_images.detach(),
                                        net_y2h(batch_cont_labels_e, batch_class_labels))

                # ---------------------- 判别器损失计算 ----------------------
                # 对应论文中原始 cGAN 损失公式：
                # L(D) = -E[log(D(x, y))] - E[log(1-D(G(z, y), y))]
                if cfg.loss_type == "vanilla":
                    real_dis_out = torch.nn.Sigmoid()(real_dis_out)
                    fake_dis_out = torch.nn.Sigmoid()(fake_dis_out)
                    d_loss_real = -torch.log(real_dis_out + 1e-20)
                    d_loss_fake = -torch.log(1 - fake_dis_out + 1e-20)
                elif cfg.loss_type == "hinge":
                    d_loss_real = torch.nn.ReLU()(1.0 - real_dis_out)
                    d_loss_fake = torch.nn.ReLU()(1.0 + fake_dis_out)
                else:
                    raise ValueError('Not supported loss type!!!')

                # 加权后求平均：这一步中利用了邻域权重（对应论文中 HVDL/SVDL 的加权求和）
                d_loss = (torch.mean(real_weights.view(-1) * d_loss_real.view(-1)) +
                          torch.mean(fake_weights.view(-1) * d_loss_fake.view(-1))) / float(
                        cfg.num_grad_acc_d)

                d_loss.backward()
            # end for grad accumulation

            optimizerD.step()
        # end for num_D_steps

        # --------------------------- 生成器训练部分 ---------------------------
        '''对应公式：L(G) = - E[ log(D(G(z,y), y) ) ]'''
        netG.train()
        optimizerG.zero_grad()

        for _ in range(cfg.num_grad_acc_g):
            # 随机采样连续标签并加噪（标签加噪机制）
            batch_cont_labels = np_rng.choice(unique_cont_labels, size=cfg.batch_size_g,
                                              replace=True)
            batch_epsilons = np_rng.normal(0, kernel_sigma, cfg.batch_size_g)
            batch_cont_labels_e = batch_cont_labels + batch_epsilons
            batch_cont_labels_e = torch.from_numpy(batch_cont_labels_e).type(torch.float).to(device)
            # 随机采样类别标签
            batch_class_labels = np_rng.choice(unique_class_labels, size=cfg.batch_size_g,
                                               replace=True)
            batch_class_labels = torch.from_numpy(batch_class_labels).type(torch.long).to(device)

            # 生成假图像，条件输入经过 net_y2h 映射
            z = torch.randn(cfg.batch_size_g, cfg.dim_gan, dtype=torch.float).to(device)
            batch_fake_images = netG(z, net_y2h(batch_cont_labels_e, batch_class_labels))

            # 判别器对假图像的输出
            if cfg.use_DiffAugment:
                dis_out = netD(DiffAugment(batch_fake_images, policy=cfg.policy),
                               net_y2h(batch_cont_labels_e, batch_class_labels))
            else:
                dis_out = netD(batch_fake_images, net_y2h(batch_cont_labels_e, batch_class_labels))

            # 生成器损失计算
            if cfg.loss_type == "vanilla":
                dis_out = torch.nn.Sigmoid()(dis_out)
                g_loss = -torch.mean(torch.log(dis_out + 1e-20))
            elif cfg.loss_type == "hinge":
                g_loss = -dis_out.mean()

            g_loss = g_loss / float(cfg.num_grad_acc_g)
            g_loss.backward()
        # end for grad accumulation for generator

        optimizerG.step()

        # --------------------------- 日志,采样,模型保存 ---------------------------
        # 打印日志：每20次迭代打印一次当前损失、真实/假样本判别概率、时间等信息
        if (niter + 1) % 20 == 0:
            print(f"CcGAN,{cfg.gan_arch}: [Iter {niter + 1}/{cfg.n_iters}] "
                  f"[D loss: {d_loss.item():.4e}] [G loss: {g_loss.item():.4e}] "
                  f"[real prob: {real_dis_out.mean().item():.3f}] "
                  f"[fake prob: {fake_dis_out.mean().item():.3f}] "
                  f"[Time: {timeit.default_timer() - start_time:.4f}]")

        """主体训练往往过程很长,为提高灵活性,这里定义一些trap事件，根据信号执行操作"""
        # 接收到"10"信号,立即采样一次并重置信号量
        # or正常采样,每隔一定迭代次数生成可视化图像（利用固定的 z_fixed 与 y_cont_fixed,y_class_fixed）
        if get_s1() or ((niter + 1) % cfg.visualize_freq == 0):
            netG.eval()
            with torch.no_grad():
                gen_imgs = netG(z_fixed, net_y2h(y_cont_fixed, y_class_fixed))
                gen_imgs = gen_imgs.detach().cpu()
                img = img_with_sep(gen_imgs.data, nrow, 1, width=2)
                save_image(img, os.path.join(images_in_train_folder, f'{niter + 1}.png'),
                           normalize=True)  # 注意: nrow表示每行摆放数量,所以=col
            # 加个锁更好 todo
            if get_s1():
                print(f"take a sample. iter {niter + 1}.")
                switch_s1(0)

        # 接收到"12"信号,立即保存模型一次并重置信号量
        if get_s2():
            print(f"saving ckpt {niter + 1}...")
            save_model(ckpts_in_train_folder, niter, netG, netD, optimizerG, optimizerD,
                       torch.get_rng_state())
            print("save successfully.")
            switch_s2(0)
        # 正常保存模型 checkpoint
        if (niter + 1) % cfg.save_n_iters_freq == 0 or (niter + 1) == cfg.n_iters:
            save_model(ckpts_in_train_folder, niter, netG, netD, optimizerG, optimizerD,
                       torch.get_rng_state())
    # end for niter

    return netG, netD


def save_model(save_folder, niter, netG, netD, optimizerG, optimizerD, rng_state):
    save_file = os.path.join(save_folder, f"ckpt_niter_{niter + 1}.pth")
    torch.save({
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'rng_state': rng_state.cpu()
    }, save_file)
