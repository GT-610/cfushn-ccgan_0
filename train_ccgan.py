# -*- coding: utf-8 -*-
# @Time    : 2025/2/12 13:16
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

import os
import timeit

from torchvision.utils import save_image

from DiffAugment_pytorch import DiffAugment
from config.constants import *
from utils.utils import SimpleProgressBar

''' 
=============================================================================
数据增强与归一化函数
--------------------------------------------------------------------------------
hflip_images: 对图像进行水平翻转（随机选择部分样本翻转）
normalize_images: 将图像归一化到 [-1,1]（用于 GAN 训练）
============================================================================= '''


def hflip_images(batch_images):
    """ 对 numpy 数组实现随机水平翻转 """
    uniform_threshold = np.random.uniform(0, 1, len(batch_images))
    indx_gt = np.nonzero(uniform_threshold > 0.5)[0]
    batch_images[indx_gt] = np.flip(batch_images[indx_gt], axis=3)
    return batch_images


def normalize_images(batch_images):
    batch_images = batch_images / 255.0
    batch_images = (batch_images - 0.5) / 0.5
    return batch_images


''' 
=============================================================================
训练函数 train_ccgan
--------------------------------------------------------------------------------
该函数实现了 ccGAN 的训练流程，核心思想包括：
1. 标签加噪机制：给真实标签加上高斯噪声（对应公式中的 ε ~ N(0, σ²)）
2. 邻域采样：利用硬邻域或软邻域方法，从训练集中选择标签处于目标标签附近的样本，
   对应于公式中的指示函数 1{|y_j^r + ε - y_i^r| ≤ κ}（硬邻域）或权重函数 exp(-v*(y_i^r - (y_j^r+ε))²)（软邻域）
3. 标签嵌入：通过 net_y2h 将一维回归标签映射到高维表示（“新型回归标签输入机制”）
4. 判别器和生成器的损失分别依据原始 cGAN 损失（vanilla 或 hinge）并结合邻域权重加权
============================================================================= '''


def train_ccgan(kernel_sigma, kappa, train_images, train_labels, netG, netD, net_y2h,
        save_images_folder, save_models_folder=None, clip_label=False):
    """
    训练连续条件GAN (ccGAN) 模型

    该函数实现了ccGAN的训练流程，主要包括：
      - 对回归标签添加高斯噪声（标签加噪机制），使得模型能够估计未出现标签处的条件分布；
      - 根据带噪目标标签在训练集中的邻域内选择真实样本（硬/软邻域方法）；
      - 根据邻域信息采样生成假标签，用于生成器生成假图像；
      - 使用标签嵌入网络 (net_y2h) 将一维回归标签映射到高维表示，并将其作为条件输入到生成器和判别器中；
      - 根据不同的损失类型（vanilla 或 hinge）计算判别器与生成器的损失，并利用邻域权重对损失进行加权；
      - 可选地使用 DiffAugment 数据增强来提升训练鲁棒性；
      - 定期保存生成图像以及模型的检查点。
    :param kernel_sigma: 高斯噪声标准差 σ（回归标签加噪）
            该机制实现了标签加噪 (ε ~ N(0, kernel_sigma²))，从而使得即使训练集中没有某个精确标签，
            也可以通过其邻域内的样本来近似估计条件分布。
    :param kappa: 邻域参数，既作为硬邻域的阈值，也在软邻域中作为高斯衰减参数
    :param train_images: 训练图像数组，像素值原始为 [0,255]，后续归一化到 [-1,1]
    :param train_labels: 与训练图像对应的回归标签数组，用作生成器与判别器的条件信息。
    :param net_y2h: 标签嵌入网络，将一维回归标签映射到高维特征空间，
            以便在生成器与判别器中更好地融合条件信息（如条件批归一化或标签投影）。
    :param clip_label: 标志位，指示是否将添加噪声后的标签进行剪裁（例如限制在 [0,1] 范围内）。
            当回归标签的取值域受限时，可启用此选项以确保标签落在合理范围内。
            默认值为 False。
    """
    # 将网络移动到 GPU，并设置 net_y2h 为评估模式（用于标签映射）
    netG = netG.to(device)
    netD = netD.to(device)
    net_y2h = net_y2h.to(device)
    net_y2h.eval()

    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(0.5, 0.999), weight_decay=0)
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(0.5, 0.999), weight_decay=0)

    # 如果从中断处恢复训练，则加载 checkpoint（与常规 GAN 训练一致）
    if save_models_folder is not None and resume_niters > 0:
        save_file = os.path.join(save_models_folder, "ckpts_in_train",
                                 f"ckpt_niter_{resume_niters}.pth")
        checkpoint = torch.load(save_file, map_location=device)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])

    # 获取所有唯一的训练标签，用于随机采样
    unique_train_labels = np.sort(np.array(list(set(train_labels))))

    # 固定噪声 z_fixed 和固定标签 y_fixed 用于训练过程中定期可视化生成效果
    n_row = 10
    n_col = n_row
    z_fixed = torch.randn(n_row * n_col, dim_gan, dtype=torch.float).to(device)
    start_label = np.quantile(train_labels, 0.05)
    end_label = np.quantile(train_labels, 0.95)
    selected_labels = np.linspace(start_label, end_label, num=n_row)
    y_fixed = np.zeros(n_row * n_col)
    for i in range(n_row):
        curr_label = selected_labels[i]
        for j in range(n_col):
            y_fixed[i * n_col + j] = curr_label
    print("Fixed labels for visualization:", y_fixed)
    y_fixed = torch.from_numpy(y_fixed).type(torch.float).view(-1, 1).to(device)

    g_loss = None
    d_loss = None
    real_dis_out = None
    fake_dis_out = None
    start_time = timeit.default_timer()

    for niter in range(resume_niters, niters):
        ''' 
        ============================================================
        判别器训练部分（对应公式中的判别器经验风险：真实项和生成项）
        -----------------------------------------------------------------
        对于每个判别器更新步骤：
          1. 从训练集中随机抽取目标标签，并对标签加噪（标签加噪机制，对应 ε ~ N(0, σ²)）
          2. 根据目标标签在训练集中寻找邻域内的真实样本：
             - 若 threshold_type 为 "hard"，利用指示函数 { |y - y_target| ≤ κ }（硬邻域公式）
             - 若为 "soft"，则通过条件 (y - y_target)² ≤ -log(threshold)/κ 选择（隐含高斯权重）
          3. 为生成器生成假标签：在目标标签邻域内均匀采样（与邻域思想一致）
          4. 利用 net_y2h 将一维标签映射到高维空间（标签嵌入）
          5. 计算判别器损失：
             - 若 loss_type 为 "vanilla"，使用 -log(D(x, y)) 和 -log(1-D(G(z,y), y))
             - 若 loss_type 为 "hinge"，使用 hinge 损失
          6. 对损失加权：软邻域时使用高斯权重（exp(-kappa*(y - y_target)²)），硬邻域时权重均为1
        ============================================================ '''
        for _ in range(num_d_steps):
            optimizerD.zero_grad()

            # 梯度累积（多步更新平均梯度）
            for _ in range(num_grad_acc_d):
                # 从唯一训练标签中随机采样 batch_size_disc 个标签
                batch_target_labels_in_dataset = np.random.choice(unique_train_labels,
                                                                  size=batch_size_d,
                                                                  replace=True)
                # 标签加噪：对每个标签加上 Gaussian 噪声，模拟公式中的 y_target + ε
                batch_epsilons = np.random.normal(0, kernel_sigma, batch_size_d)
                batch_target_labels = batch_target_labels_in_dataset + batch_epsilons

                # 初始化数组：存放选中的真实样本索引及生成假标签
                batch_real_indx = np.zeros(batch_size_d, dtype=int)
                batch_fake_labels = np.zeros(batch_size_d)

                # 对于每个目标标签，寻找其邻域内的真实样本，并生成假标签
                for j in range(batch_size_d):
                    if threshold_type == "hard":
                        # 硬邻域：选择满足 |train_label - target_label| ≤ κ 的样本
                        indx_real_in_vicinity = \
                            np.nonzero(np.abs(train_labels - batch_target_labels[j]) <= kappa)[0]
                    else:
                        # 软邻域：利用逆向条件 (y - target)² ≤ -log(threshold)/κ
                        indx_real_in_vicinity = np.nonzero(
                                (train_labels - batch_target_labels[j]) ** 2 <= -np.log(
                                        nonzero_soft_weight_threshold) / kappa)[0]

                    # 如果当前目标标签在训练集中无对应邻域样本，则重新采样（保证至少1个样本）
                    while len(indx_real_in_vicinity) < 1:
                        batch_epsilons_j = np.random.normal(0, kernel_sigma, 1)
                        batch_target_labels[j] = batch_target_labels_in_dataset[
                                                     j] + batch_epsilons_j
                        if clip_label:
                            batch_target_labels = np.clip(batch_target_labels, 0.0, 1.0)
                        if threshold_type == "hard":
                            indx_real_in_vicinity = \
                                np.nonzero(np.abs(train_labels - batch_target_labels[j]) <= kappa)[
                                    0]
                        else:
                            indx_real_in_vicinity = np.nonzero(
                                    (train_labels - batch_target_labels[j]) ** 2 <= -np.log(
                                            nonzero_soft_weight_threshold) / kappa)[0]

                    # 随机从邻域内选取一个真实样本的索引（对应于利用邻域内样本联合估计条件分布）
                    batch_real_indx[j] = np.random.choice(indx_real_in_vicinity, size=1)[0]

                    # 为生成器生成假标签：在目标标签邻域内均匀采样
                    if threshold_type == "hard":
                        lb = batch_target_labels[j] - kappa
                        ub = batch_target_labels[j] + kappa
                    else:
                        # 软邻域使用区间 [target - sqrt(-log(threshold)/κ), target + sqrt(-log(threshold)/κ)]
                        lb = batch_target_labels[j] - np.sqrt(
                                -np.log(nonzero_soft_weight_threshold) / kappa)
                        ub = batch_target_labels[j] + np.sqrt(
                                -np.log(nonzero_soft_weight_threshold) / kappa)
                    lb = max(0.0, lb)
                    ub = min(ub, 1.0)
                    batch_fake_labels[j] = np.random.uniform(lb, ub, size=1)[0]
                # end for j

                # ---------------------- 获取真实样本 -------------------------
                # 根据选中的索引，从训练集中抽取真实图像和标签，
                # 并做数据增强（水平翻转）和归一化（对应 GAN 输入要求）
                batch_real_images = torch.from_numpy(
                        normalize_images(hflip_images(train_images[batch_real_indx])))
                batch_real_images = batch_real_images.type(torch.float).to(device)
                batch_real_labels = train_labels[batch_real_indx]
                batch_real_labels = torch.from_numpy(batch_real_labels).type(torch.float).to(device)

                # ---------------------- 生成假样本 ---------------------------
                # 将生成假标签转换为 tensor，并利用标签嵌入网络 net_y2h（新型回归标签输入机制）
                batch_fake_labels = torch.from_numpy(batch_fake_labels).type(torch.float).to(device)
                z = torch.randn(batch_size_d, dim_gan, dtype=torch.float).to(device)
                batch_fake_images = netG(z, net_y2h(batch_fake_labels))

                # 将目标标签（用于判别器条件输入）转换到 GPU
                batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).to(
                    device)

                # ---------------------- 计算邻域权重 -------------------------
                # 若使用软邻域，权重根据 exp(-kappa*(y - y_target)²) 计算，
                # 对应公式中软邻域权重 w(y_i, y) = exp(-v*(y_i - y)²)
                if threshold_type == "soft":
                    real_weights = torch.exp(
                            -kappa * (batch_real_labels - batch_target_labels) ** 2).to(device)
                    fake_weights = torch.exp(
                            -kappa * (batch_fake_labels - batch_target_labels) ** 2).to(device)
                else:
                    # 硬邻域时，每个样本权重均为1（已隐式在样本筛选中体现）
                    real_weights = torch.ones(batch_size_d, dtype=torch.float).to(device)
                    fake_weights = torch.ones(batch_size_d, dtype=torch.float).to(device)
                # end if threshold type

                # ---------------------- 判别器前向传播 ----------------------
                # 利用 DiffAugment 进行数据增强（可选）
                if use_DiffAugment:
                    real_dis_out = netD(DiffAugment(batch_real_images, policy=policy),
                                        net_y2h(batch_target_labels))
                    fake_dis_out = netD(DiffAugment(batch_fake_images.detach(), policy=policy),
                                        net_y2h(batch_target_labels))
                else:
                    real_dis_out = netD(batch_real_images, net_y2h(batch_target_labels))
                    fake_dis_out = netD(batch_fake_images.detach(), net_y2h(batch_target_labels))

                # ---------------------- 判别器损失计算 ----------------------
                # 对应论文中原始 cGAN 损失公式：
                # L(D) = -E[log(D(x, y))] - E[log(1-D(G(z, y), y))]
                if loss_type == "vanilla":
                    real_dis_out = torch.nn.Sigmoid()(real_dis_out)
                    fake_dis_out = torch.nn.Sigmoid()(fake_dis_out)
                    d_loss_real = -torch.log(real_dis_out + 1e-20)
                    d_loss_fake = -torch.log(1 - fake_dis_out + 1e-20)
                elif loss_type == "hinge":
                    d_loss_real = torch.nn.ReLU()(1.0 - real_dis_out)
                    d_loss_fake = torch.nn.ReLU()(1.0 + fake_dis_out)
                else:
                    raise ValueError('Not supported loss type!!!')

                # 加权后求平均：这一步中利用了邻域权重（对应论文中 HVDL/SVDL 的加权求和）
                d_loss = (torch.mean(real_weights.view(-1) * d_loss_real.view(-1)) +
                          torch.mean(fake_weights.view(-1) * d_loss_fake.view(-1))) / float(
                        num_grad_acc_d)

                d_loss.backward()
            # end for grad accumulation

            optimizerD.step()
        # end for num_D_steps

        ''' ============================================================
        生成器训练部分（对应公式：L(G) = - E[ log(D(G(z,y), y) ) ]）
        -----------------------------------------------------------------
        1. 随机采样目标标签并加噪
        2. 利用生成器 netG 生成假图像（条件输入为 net_y2h(batch_target_labels)）
        3. 计算生成器损失（vanilla 或 hinge 损失）
        ============================================================ '''
        netG.train()
        optimizerG.zero_grad()
        for _ in range(num_grad_acc_g):
            # 随机采样目标标签并加噪（标签加噪机制）
            batch_target_labels_in_dataset = np.random.choice(unique_train_labels,
                                                              size=batch_size_g, replace=True)
            batch_epsilons = np.random.normal(0, kernel_sigma, batch_size_g)
            batch_target_labels = batch_target_labels_in_dataset + batch_epsilons
            batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).to(device)

            # 生成假图像，条件输入经过 net_y2h 映射
            z = torch.randn(batch_size_g, dim_gan, dtype=torch.float).to(device)
            batch_fake_images = netG(z, net_y2h(batch_target_labels))

            # 判别器对假图像的输出
            if use_DiffAugment:
                dis_out = netD(DiffAugment(batch_fake_images, policy=policy),
                               net_y2h(batch_target_labels))
            else:
                dis_out = netD(batch_fake_images, net_y2h(batch_target_labels))

            # 生成器损失计算
            if loss_type == "vanilla":
                dis_out = torch.nn.Sigmoid()(dis_out)
                g_loss = -torch.mean(torch.log(dis_out + 1e-20))
            elif loss_type == "hinge":
                g_loss = -dis_out.mean()

            g_loss = g_loss / float(num_grad_acc_g)
            g_loss.backward()
        # end for grad accumulation for generator

        optimizerG.step()

        # 打印日志：每20次迭代打印一次当前损失、真实/假样本判别概率、时间等信息
        if (niter + 1) % 20 == 0:
            print(f"CcGAN,{GAN_arch}: [Iter {niter + 1}/{niters}] [D loss: {d_loss.item():.4e}] "
                  f"[G loss: {g_loss.item():.4e}] [real prob: {real_dis_out.mean().item():.3f}] "
                  f"[fake prob: {fake_dis_out.mean().item():.3f}] [Time: {timeit.default_timer() - start_time:.4f}]")

        # 每 visualize_freq 次迭代生成可视化图像（利用固定的 z_fixed 与 y_fixed）
        if (niter + 1) % visualize_freq == 0:
            netG.eval()
            with torch.no_grad():
                gen_imgs = netG(z_fixed, net_y2h(y_fixed))
                gen_imgs = gen_imgs.detach().cpu()
                save_image(gen_imgs.data, os.path.join(save_images_folder, f'{niter + 1}.png'),
                           nrow=n_row, normalize=True)

        # 保存模型 checkpoint
        if save_models_folder is not None and (
                (niter + 1) % save_niters_freq == 0 or (niter + 1) == niters):
            save_file = os.path.join(save_models_folder, "ckpts_in_train",
                                     f"ckpt_niter_{niter + 1}.pth")
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'rng_state': torch.get_rng_state()
            }, save_file)
    # end for niter

    return netG, netD


''' 
=============================================================================
采样函数 sample_ccgan_given_labels
--------------------------------------------------------------------------------
该函数用于利用训练好的生成器 netG，根据给定的连续标签采样生成图像，
同样利用 net_y2h 将一维标签映射到高维。采样过程不涉及梯度计算，只用于推理。
============================================================================= '''


def sample_ccgan_given_labels(netG, net_y2h, labels, batch_size=500, to_numpy=True, denorm=True,
        verbose=True):
    nfake = len(labels)
    if batch_size > nfake:
        batch_size = nfake

    fake_images = []
    fake_labels = np.concatenate((labels, labels[0:batch_size]))
    netG = netG.to(device)
    netG.eval()
    net_y2h = net_y2h.to(device)
    net_y2h.eval()
    with torch.no_grad():
        if verbose:
            pb = SimpleProgressBar()
        n_img_got = 0
        while n_img_got < nfake:
            z = torch.randn(batch_size, dim_gan, dtype=torch.float).to(device)
            y = torch.from_numpy(fake_labels[n_img_got:(n_img_got + batch_size)]).type(
                    torch.float).view(-1, 1).to(device)
            batch_fake_images = netG(z, net_y2h(y))
            if denorm:
                # 将生成图像从 [-1,1] 映射回 [0,255]
                assert batch_fake_images.max().item() <= 1.0 and batch_fake_images.min().item() >= -1.0
                batch_fake_images = batch_fake_images * 0.5 + 0.5
                batch_fake_images = batch_fake_images * 255.0
                batch_fake_images = batch_fake_images.type(torch.uint8)
            fake_images.append(batch_fake_images.cpu())
            n_img_got += batch_size
            if verbose:
                pb.update(min(float(n_img_got) / nfake, 1) * 100)
        # end while

    fake_images = torch.cat(fake_images, dim=0)
    fake_images = fake_images[0:nfake]  # 去除多余部分
    fake_labels = fake_labels[0:nfake]

    if to_numpy:
        fake_images = fake_images.numpy()
    else:
        fake_labels = torch.from_numpy(fake_labels)

    return fake_images, fake_labels
