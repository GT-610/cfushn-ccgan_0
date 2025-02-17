# -*- coding: utf-8 -*-
# @Time    : 2025/2/12 13:16
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

import timeit

from torchvision.utils import save_image

from DiffAugment_pytorch import DiffAugment
from config.config import *
from utils.utils2 import normalize_images, hflip_images

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


def train_ccgan(kernel_sigma, kappa, images, cont_labels, class_labels, netG, netD, net_y2h,
        save_images_folder, save_models_folder=None, clip_label=False):
    # 将网络移动到 GPU，并设置 net_y2h 为评估模式（用于标签映射）
    netG = netG.to(device)
    netD = netD.to(device)
    net_y2h = net_y2h.to(device)
    net_y2h.eval()

    optimizerG = torch.optim.Adam(netG.parameters(), lr=LR_G, betas=(0.5, 0.999), weight_decay=0)
    optimizerD = torch.optim.Adam(netD.parameters(), lr=LR_D, betas=(0.5, 0.999), weight_decay=0)

    # 如果从中断处恢复训练，则加载 checkpoint（与常规 GAN 训练一致）
    if save_models_folder is not None and RESUME_N_ITERS > 0:
        save_file = os.path.join(save_models_folder, "ckpts_in_train",
                                 f"ckpt_niter_{RESUME_N_ITERS}.pth")
        checkpoint = torch.load(save_file, map_location=device)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])

    # 获取所有唯一的训练标签，用于随机采样
    unique_cont_labels = np.sort(np.array(list(set(cont_labels))))
    unique_class_labels = np.sort(np.array(list(set(class_labels))))

    # 固定噪声 z_fixed 和固定标签 y_cont_fixed,y_class_fixed 用于训练过程中定期可视化生成效果
    n_row, n_col = 10, 25  # 每行年龄依次增大,每5列一个人种
    z_fixed = torch.randn(n_row * n_col, DIM_GAN, dtype=torch.float).to(device)
    # 分位数, eg:q=0.5就是中位数
    start_label = np.quantile(cont_labels, 0.05)
    end_label = np.quantile(cont_labels, 0.95)
    # 在线性空间中，生成某区间等距分布的num个数
    selected_cont_labels = np.linspace(start_label, end_label, num=n_row)
    selected_class_labels = unique_class_labels  # class类别数不多, 直接全用了
    y_cont_fixed = np.zeros(n_row * n_col)
    y_class_fixed = np.zeros(n_row * n_col)
    for i in range(n_row):
        curr_label = selected_cont_labels[i]
        # 逐行递增年龄标签
        for j in range(n_col):
            y_cont_fixed[i * n_col + j] = curr_label
            y_class_fixed[i * n_col + j] = selected_class_labels[j // (n_col // NUM_CLASSES)]

    print(f"Fixed labels for visualization (covering all classes): {y_cont_fixed} ")
    y_cont_fixed = torch.from_numpy(y_cont_fixed).type(torch.float).view(-1, 1).to(device)
    y_class_fixed = torch.from_numpy(y_class_fixed).type(torch.long).view(-1).to(device)

    # ## 按类别将所有样本下标分组
    # # index_by_class = [np.where(class_labels == k)[0] for k in range(num_classes)]
    # # 等价写法:
    # N = len(images)  # 样本数
    # index_by_class = [[] for _ in range(num_classes)]
    # for i in range(N):
    #     c = class_labels[i]
    #     index_by_class[c].append(i)
    # # # 如果需要，将其转成 np.array
    # # for k in range(num_classes):
    # #     index_by_class[k] = np.array(index_by_class[k], dtype=int)

    g_loss = None
    d_loss = None
    real_dis_out = None
    fake_dis_out = None
    start_time = timeit.default_timer()

    for niter in range(RESUME_N_ITERS, N_ITERS):

        ''' 
        ============================================================
        判别器训练部分
        ============================================================ '''
        for _ in range(NUM_D_STEPS):
            optimizerD.zero_grad()

            # 梯度累积（多步更新平均梯度）
            for _ in range(NUM_GRAD_ACC_D):
                # 从连续标签(唯一)中随机采样 batch_size 个标签
                batch_cont_labels = np.random.choice(unique_cont_labels, size=BATCH_SIZE_D,
                                                     replace=True)
                batch_fake_cont_labels = np.zeros(BATCH_SIZE_D)
                # 类别标签不再进行抽取,因为后续选出真实样本后,三者(img,cont,class)需要对应上
                # 具体流程:先对当前连续标签加噪再进行邻域操作,然后在邻域内随机选一个真实样本
                # (假设不同类别的样本在各个连续标签都很充足, 在每次的邻域择取中出现的概率差不多)
                # batch_class_labels = np.zeros(batch_size_d)

                # 对连续标签加噪：对每个标签加上 Gaussian 噪声，模拟公式中的 y_target + ε
                batch_epsilons = np.random.normal(0, kernel_sigma, BATCH_SIZE_D)
                batch_cont_labels_e = batch_cont_labels + batch_epsilons

                # 初始化数组：存放选中的真实样本索引
                batch_real_index = np.zeros(BATCH_SIZE_D, dtype=int)

                # 对于每个目标标签，寻找其邻域内的真实样本，并生成假标签
                for j in range(BATCH_SIZE_D):
                    if THRESHOLD_TYPE == "hard":
                        # 硬邻域：选择满足 |train_label - target_label| ≤ κ 的样本
                        index_real_in_vicinity = \
                            np.nonzero(np.abs(cont_labels - batch_cont_labels_e[j]) <= kappa)[0]
                    else:
                        # 软邻域：利用逆向条件 (y - target)² ≤ -log(threshold)/κ
                        index_real_in_vicinity = np.nonzero(
                                (cont_labels - batch_cont_labels_e[j]) ** 2 <= -np.log(
                                        NONZERO_SOFT_WEIGHT_THRESHOLD) / kappa)[0]

                    # 如果当前目标标签在训练集中无对应邻域样本，则重新采样（保证至少1个样本）
                    while len(index_real_in_vicinity) < 1:
                        batch_epsilons_j = np.random.normal(0, kernel_sigma, 1)
                        batch_cont_labels_e[j] = batch_cont_labels[j] + batch_epsilons_j
                        if clip_label:
                            batch_cont_labels_e = np.clip(batch_cont_labels_e, 0.0, 1.0)
                        if THRESHOLD_TYPE == "hard":
                            index_real_in_vicinity = \
                                np.nonzero(np.abs(cont_labels - batch_cont_labels_e[j]) <= kappa)[0]
                        else:
                            index_real_in_vicinity = np.nonzero(
                                    (cont_labels - batch_cont_labels_e[j]) ** 2 <= -np.log(
                                            NONZERO_SOFT_WEIGHT_THRESHOLD) / kappa)[0]

                    # 随机从邻域内选取一个真实样本的索引（对应于利用邻域内样本联合估计条件分布）
                    batch_real_index[j] = np.random.choice(index_real_in_vicinity, size=1)[0]

                    # 为生成器生成假标签：在目标标签邻域内均匀采样
                    if THRESHOLD_TYPE == "hard":
                        lb = batch_cont_labels_e[j] - kappa
                        ub = batch_cont_labels_e[j] + kappa
                    else:
                        # 软邻域使用区间 [target - sqrt(-log(threshold)/κ), target + sqrt(-log(threshold)/κ)]
                        lb = batch_cont_labels_e[j] - np.sqrt(
                                -np.log(NONZERO_SOFT_WEIGHT_THRESHOLD) / kappa)
                        ub = batch_cont_labels_e[j] + np.sqrt(
                                -np.log(NONZERO_SOFT_WEIGHT_THRESHOLD) / kappa)
                    # 裁剪一下
                    lb = max(0.0, lb)
                    ub = min(ub, 1.0)
                    # 在lb~ub的均匀分布中, 生成size=1的采样结果, 取结果数组的首个
                    batch_fake_cont_labels[j] = np.random.uniform(lb, ub, size=1)[0]

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
                z = torch.randn(BATCH_SIZE_D, DIM_GAN, dtype=torch.float).to(device)
                batch_fake_images = netG(z, net_y2h(batch_fake_cont_labels, batch_class_labels))

                # 将目标标签（用于判别器条件输入）转换到 GPU
                batch_cont_labels_e = torch.from_numpy(batch_cont_labels_e).type(torch.float).to(
                        device)
                # todo: 无法得到与batch_cont_labels_e对应的真实类别标签,只能使用batch_real_labels对应的?

                # ---------------------- 计算邻域权重 -------------------------
                # 若使用软邻域，权重根据 exp(-kappa*(y - y_target)²) 计算，
                # 对应公式中软邻域权重 w(y_i, y) = exp(-v*(y_i - y)²)
                if THRESHOLD_TYPE == "soft":
                    real_weights = torch.exp(
                            -kappa * (batch_real_labels - batch_cont_labels_e) ** 2).to(device)
                    fake_weights = torch.exp(
                            -kappa * (batch_fake_cont_labels - batch_cont_labels_e) ** 2).to(device)
                else:
                    # 硬邻域时，每个样本权重均为1（已隐式在样本筛选中体现）
                    real_weights = torch.ones(BATCH_SIZE_D, dtype=torch.float).to(device)
                    fake_weights = torch.ones(BATCH_SIZE_D, dtype=torch.float).to(device)
                # end if threshold type

                # ---------------------- 判别器前向传播 ----------------------
                # 利用 DiffAugment 进行数据增强（可选）
                if USE_DiffAugment:
                    real_dis_out = netD(DiffAugment(batch_real_images, policy=POLICY),
                                        net_y2h(batch_cont_labels_e, batch_class_labels))
                    fake_dis_out = netD(DiffAugment(batch_fake_images.detach(), policy=POLICY),
                                        net_y2h(batch_cont_labels_e, batch_class_labels))
                else:
                    real_dis_out = netD(batch_real_images,
                                        net_y2h(batch_cont_labels_e, batch_class_labels))
                    fake_dis_out = netD(batch_fake_images.detach(),
                                        net_y2h(batch_cont_labels_e, batch_class_labels))

                # ---------------------- 判别器损失计算 ----------------------
                # 对应论文中原始 cGAN 损失公式：
                # L(D) = -E[log(D(x, y))] - E[log(1-D(G(z, y), y))]
                if LOSS_TYPE == "vanilla":
                    real_dis_out = torch.nn.Sigmoid()(real_dis_out)
                    fake_dis_out = torch.nn.Sigmoid()(fake_dis_out)
                    d_loss_real = -torch.log(real_dis_out + 1e-20)
                    d_loss_fake = -torch.log(1 - fake_dis_out + 1e-20)
                elif LOSS_TYPE == "hinge":
                    d_loss_real = torch.nn.ReLU()(1.0 - real_dis_out)
                    d_loss_fake = torch.nn.ReLU()(1.0 + fake_dis_out)
                else:
                    raise ValueError('Not supported loss type!!!')

                # 加权后求平均：这一步中利用了邻域权重（对应论文中 HVDL/SVDL 的加权求和）
                d_loss = (torch.mean(real_weights.view(-1) * d_loss_real.view(-1)) +
                          torch.mean(fake_weights.view(-1) * d_loss_fake.view(-1))) / float(
                        NUM_GRAD_ACC_D)

                d_loss.backward()
            # end for grad accumulation

            optimizerD.step()
        # end for num_D_steps

        ''' 
        ============================================================
        生成器训练部分（对应公式：L(G) = - E[ log(D(G(z,y), y) ) ]）
        -----------------------------------------------------------------
        1. 随机采样目标标签并加噪
        2. 利用生成器 netG 生成假图像（条件输入为 net_y2h(batch_cont_labels_e,batch_class_labels)）
        3. 计算生成器损失（vanilla 或 hinge 损失）
        ============================================================ '''
        netG.train()
        optimizerG.zero_grad()
        for _ in range(NUM_GRAD_ACC_G):
            # 随机采样目标标签并加噪（标签加噪机制）
            batch_cont_labels = np.random.choice(unique_cont_labels, size=BATCH_SIZE_G,
                                                 replace=True)
            batch_epsilons = np.random.normal(0, kernel_sigma, BATCH_SIZE_G)
            batch_cont_labels_e = batch_cont_labels + batch_epsilons
            batch_cont_labels_e = torch.from_numpy(batch_cont_labels_e).type(torch.float).to(device)
            # 随机采样类别标签
            batch_class_labels = np.random.choice(unique_class_labels, size=BATCH_SIZE_G,
                                                  replace=True)
            batch_class_labels = torch.from_numpy(batch_class_labels).type(torch.long).to(device)

            # 生成假图像，条件输入经过 net_y2h 映射
            z = torch.randn(BATCH_SIZE_G, DIM_GAN, dtype=torch.float).to(device)
            batch_fake_images = netG(z, net_y2h(batch_cont_labels_e, batch_class_labels))

            # 判别器对假图像的输出
            if USE_DiffAugment:
                dis_out = netD(DiffAugment(batch_fake_images, policy=POLICY),
                               net_y2h(batch_cont_labels_e, batch_class_labels))
            else:
                dis_out = netD(batch_fake_images, net_y2h(batch_cont_labels_e, batch_class_labels))

            # 生成器损失计算
            if LOSS_TYPE == "vanilla":
                dis_out = torch.nn.Sigmoid()(dis_out)
                g_loss = -torch.mean(torch.log(dis_out + 1e-20))
            elif LOSS_TYPE == "hinge":
                g_loss = -dis_out.mean()

            g_loss = g_loss / float(NUM_GRAD_ACC_G)
            g_loss.backward()
        # end for grad accumulation for generator

        optimizerG.step()

        # 打印日志：每20次迭代打印一次当前损失、真实/假样本判别概率、时间等信息
        if (niter + 1) % 20 == 0:
            print(f"CcGAN,{GAN_ARCH}: [Iter {niter + 1}/{N_ITERS}] [D loss: {d_loss.item():.4e}] "
                  f"[G loss: {g_loss.item():.4e}] [real prob: {real_dis_out.mean().item():.3f}] "
                  f"[fake prob: {fake_dis_out.mean().item():.3f}] [Time: {timeit.default_timer() - start_time:.4f}]")

        # 每 visualize_freq 次迭代生成可视化图像（利用固定的 z_fixed 与 y_cont_fixed,y_class_fixed）
        if (niter + 1) % VISUALIZE_FREQ == 0:
            netG.eval()
            with torch.no_grad():
                gen_imgs = netG(z_fixed, net_y2h(y_cont_fixed, y_class_fixed))
                gen_imgs = gen_imgs.detach().cpu()
                save_image(gen_imgs.data, os.path.join(save_images_folder, f'{niter + 1}.png'),
                           nrow=n_row, normalize=True)

        # 保存模型 checkpoint
        if save_models_folder is not None and (
                (niter + 1) % SAVE_N_ITERS_FREQ == 0 or (niter + 1) == N_ITERS):
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
