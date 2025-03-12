from models.resnet_x2y import *
from models.resnet_y2h import ResNetY2H
from models.sngan import *
from utils.utils import *
from .train_ccgan import train_ccgan
from .train_embed import *

device = cfg.device
np_rng = np.random.default_rng(cfg.seed)


def train_process(data):
    images, cont_labels, class_labels = data

    # -------------------- 确定GAN输出目录 --------------------
    # 目录名中有变量拼接 (cfg.gan_output_path@property方法能动态get)
    os.makedirs(cfg.gan_output_path, exist_ok=True)
    print(f'output_path:{cfg.gan_output_path}\n')

    # -------------------- 构建训练集和 DataLoader --------------------
    # 这里假设 ImgsDataset 类已经支持同时接受三个参数：图像、连续标签、离散标签，并自动归一化图像
    train_set = ImgsDataset(images, cont_labels, class_labels, normalize=True)
    train_loader_embed_net = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size_embed,
                                                         shuffle=True, num_workers=cfg.num_workers)

    #######################################################################################
    '''               Pre-trained CNN and GAN for label embedding                       '''
    #######################################################################################
    # -------------------- 定义预训练模型的 checkpoint 文件名 --------------------
    path_to_embed_models = os.path.join(cfg.output_path, 'embed_models')  # 固定该版
    os.makedirs(path_to_embed_models, exist_ok=True)
    path_to_embed_ckpt = os.path.join(path_to_embed_models, "embed_x2y_ckpt_in_train")
    os.makedirs(path_to_embed_ckpt, exist_ok=True)
    net_embed_filename_ckpt = os.path.join(path_to_embed_models, 'ckpt_{}_epoch_{}_seed_{}.pth'
                                           .format(cfg.net_embed_type, cfg.epoch_cnn_embed,
                                                   cfg.seed))
    net_y2h_filename_ckpt = os.path.join(path_to_embed_models, 'ckpt_net_y2h_epoch_{}_seed_{}.pth'
                                         .format(cfg.epoch_net_y2h, cfg.seed))

    # -------------------- 构建图像嵌入模型 net_embed --------------------
    net_x2y = None
    if cfg.net_embed_type == "ResNet18_embed":
        net_x2y = ResNet18_x2y(dim_embed=cfg.dim_embed)
    elif cfg.net_embed_type == "ResNet34_embed":
        net_x2y = ResNet34_x2y(dim_embed=cfg.dim_embed)
    elif cfg.net_embed_type == "ResNet50_embed":
        net_x2y = ResNet50_x2y(dim_embed=cfg.dim_embed)
    net_x2y = net_x2y.to(device)

    # -------------------- 构建标签映射模型 net_y2h --------------------
    net_y2h = ResNetY2H(dim_embed=cfg.dim_embed)
    net_y2h = net_y2h.to(device)

    ## (1). 训练 net_embed：将图像映射到嵌入空间，然后通过 h2y 映射回标签（x2h+h2y）
    if not os.path.isfile(net_embed_filename_ckpt):
        if cfg.gpu_parallel:
            net_x2y = nn.DataParallel(net_x2y)
        print("Start training CNN for label embedding >>>")
        net_x2y = train_net_embed(net_x2y=net_x2y,
                                  train_loader=train_loader_embed_net, test_loader=None,
                                  epochs=cfg.epoch_cnn_embed,
                                  resume_epoch=cfg.resume_epoch_cnn_embed,
                                  lr_base=cfg.base_lr_x2y, lr_decay_factor=0.1,
                                  lr_decay_epochs=[80, 140],
                                  weight_decay=1e-4, path_to_ckpt=path_to_embed_ckpt)
        # 保存训练好的 net_embed 模型 (对于有DataParallel包裹的需要先.module)
        net_x2y = net_x2y.module if cfg.gpu_parallel else net_x2y
        torch.save({'net_state_dict': net_x2y.state_dict()}, net_embed_filename_ckpt)
    else:
        print("net_embed ckpt already exists")
        print("Loading...")
        checkpoint = torch.load(net_embed_filename_ckpt, weights_only=True, map_location=device)
        net_x2y.load_state_dict(checkpoint['net_state_dict'])
        print("Loaded successfully.\n")

    ## (2). 训练 net_y2h：将标签映射到与图像嵌入相同的空间
    if not os.path.isfile(net_y2h_filename_ckpt):
        print("Start training net_y2h >>>")
        net_y2h = train_net_y2h(cont_labels, class_labels, net_y2h, net_x2y,
                                epochs=cfg.epoch_net_y2h,
                                lr_base=cfg.base_lr_y2h, lr_decay_factor=0.1,
                                lr_decay_epochs=[150, 250, 350],
                                weight_decay=1e-4, batch_size=128)
        # 保存训练好的 net_y2h 模型
        torch.save({'net_state_dict': net_y2h.state_dict(), }, net_y2h_filename_ckpt)
    else:
        print("net_y2h ckpt already exists")
        print("Loading...")
        checkpoint = torch.load(net_y2h_filename_ckpt, weights_only=True, map_location=device)
        net_y2h.load_state_dict(checkpoint['net_state_dict'])
        print("Loaded successfully.\n")

    ## 测试一次
    test_embed(net_x2y, net_y2h, cont_labels)

    #######################################################################################
    '''                                    GAN training                                 '''
    #######################################################################################
    print("CcGAN: {}, {}, Sigma is {:.4f}, Kappa is {:.4f}.".format(
            cfg.gan_arch, cfg.threshold_type, cfg.kernel_sigma, cfg.kappa))
    save_models_folder = os.path.join(cfg.gan_output_path, 'saved_models')
    os.makedirs(save_models_folder, exist_ok=True)
    save_images_folder = os.path.join(cfg.gan_output_path, 'saved_images')
    os.makedirs(save_images_folder, exist_ok=True)
    ckpts_in_train_folder = os.path.join(save_models_folder, 'ckpts_in_train')
    os.makedirs(ckpts_in_train_folder, exist_ok=True)
    images_in_train_folder = os.path.join(save_images_folder, 'images_in_train')
    os.makedirs(images_in_train_folder, exist_ok=True)
    start = timeit.default_timer()

    print("Begin Training >>>")
    ckpt_gan_path = os.path.join(save_models_folder, f'ckpt_niter_{cfg.n_iters}.pth')
    print(ckpt_gan_path)
    netG = None
    netD = None
    if not os.path.isfile(ckpt_gan_path):
        # 根据 GAN 架构选择生成器与判别器
        if cfg.gan_arch == "SAGAN":
            # netG = sagan_generator(nz=dim_gan, dim_embed=dim_embed).to(device)
            # netD = sagan_discriminator(dim_embed=dim_embed).to(device)
            pass
        else:
            netG = SnganGenerator(nz=cfg.dim_gan, dim_embed=cfg.dim_embed).to(device)
            netD = SnganDiscriminator(dim_embed=cfg.dim_embed).to(device)
        # 使用多GPU并行训练
        if cfg.gpu_parallel:
            netG = nn.DataParallel(netG)
            netD = nn.DataParallel(netD)

        # 调用 train_ccgan 函数进行 GAN 训练
        netG, _ = train_ccgan(cfg.kernel_sigma, cfg.kappa, images, cont_labels, class_labels,
                              netG, netD, net_y2h,
                              images_in_train_folder=images_in_train_folder,
                              ckpts_in_train_folder=ckpts_in_train_folder)
        # 保存训练好的生成器模型
        torch.save({'netG_state_dict': netG.state_dict()}, ckpt_gan_path)
        stop = timeit.default_timer()
        print("GAN training finished; Time elapses: {}s".format(stop - start))
    else:
        print("Loading pre-trained generator >>>")
        checkpoint = torch.load(ckpt_gan_path, weights_only=True, map_location=device)
        # 根据 GAN 架构选择生成器
        if cfg.gan_arch == "SAGAN":
            # netG = sagan_generator(nz=dim_gan, dim_embed=dim_embed).to(device)
            pass
        else:
            netG = SnganGenerator(nz=cfg.dim_gan, dim_embed=cfg.dim_embed).to(device)
        if cfg.gpu_parallel:
            netG = nn.DataParallel(netG)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        print("Loaded successfully.\n")

    return netG, net_y2h


def test_embed(net_embed, net_y2h, cont_labels):
    # -------------------- 简单测试：标签映射是否正确 --------------------
    net_embed.eval()
    net_y2h.eval()
    # 从连续标签中随机选择 10 个用于测试映射效果
    unique_cont_labels_norm = np.sort(np.array(list(set(cont_labels))))
    index_tmp = np.arange(len(unique_cont_labels_norm))
    np_rng.shuffle(index_tmp)
    index_tmp = index_tmp[:10]  # 60个唯一标签索引打乱后,取前10个
    # labels_tmp: 形状 (10, 1)，注意这里连续标签已经归一化到 [0,1]
    labels_tmp = unique_cont_labels_norm[index_tmp].reshape(-1, 1)
    labels_tmp = torch.from_numpy(labels_tmp).type(torch.float).to(device)
    # 添加噪声，模拟连续标签的不确定性
    epsilons_tmp = np_rng.normal(0, 0.2, len(labels_tmp))
    epsilons_tmp = torch.from_numpy(epsilons_tmp).view(-1, 1).type(torch.float).to(device)
    labels_noise_tmp = torch.clamp(labels_tmp + epsilons_tmp, 0.0, 1.0)

    # 对于离散标签测试，选择一个固定的类别，例如 2（假设类别取值为 0~4）
    fixed_class = torch.full((labels_tmp.size(0),), 2, dtype=torch.long).to(device)

    # 获取预训练好的 net_embed 中的两个输出分支
    net_h2y_cont = net_embed.h2y_cont  # 用于连续标签重构
    net_h2y_class = net_embed.h2y_class  # 用于离散标签重构
    with torch.no_grad():
        # 将加噪后的连续标签与固定离散标签一起输入 net_y2h，得到联合标签嵌入 h
        h = net_y2h(labels_noise_tmp, fixed_class)
        # 利用连续分支反映射得到重构的连续标签
        labels_rec_tmp = net_h2y_cont(h).cpu().numpy().reshape(-1, 1)
        # 利用分类分支得到重构的离散标签 logits
        class_logits = net_h2y_class(h).cpu().numpy()  # 形状 (10, NUM_CLASSES)

        # 对连续标签部分不加噪声的原始标签也通过 net_y2h 进行映射重构，作为对比
        h_orig = net_y2h(labels_tmp, fixed_class)
        labels_rec_orig = net_h2y_cont(h_orig).cpu().numpy().reshape(-1, 1)

        # 同时，计算连续标签在加噪前后的差异，用于评估噪声对映射的影响
        labels_tmp_np = labels_tmp.cpu().numpy()
        labels_noise_np = labels_noise_tmp.cpu().numpy()

    # 输出测试结果：连续标签重构对比
    results1 = np.concatenate((labels_tmp_np, labels_rec_orig, labels_rec_tmp), axis=1)
    print("Continuous labels vs reconstructed labels:")
    print("Original, Rec (no noise), Rec (with noise)")
    print(results1)

    # 计算连续标签误差（均方差），作为指标
    labels_diff = (labels_tmp_np - labels_noise_np) ** 2
    # 这里也可以计算隐层表示差异等
    print("Continuous labels diff (squared):")
    print(labels_diff)

    # 输出离散标签部分的预测
    # 对 logits 做 softmax 并取最大概率对应的类别
    import torch.nn.functional as F

    predicted_class = F.softmax(torch.tensor(class_logits), dim=1).argmax(dim=1).numpy()
    print("Fixed discrete label: 2, Predicted discrete label from net_y2h fusion branch:")
    print(predicted_class, "\n")
