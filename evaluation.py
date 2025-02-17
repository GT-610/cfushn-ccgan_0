from models.eval.ResNet_class_eval import ResNet34_class_eval
from models.eval.ResNet_regre_eval import ResNet34_regre_eval
from models.eval.autoencoder import encoder
from utils.utils import *

print("\n========================================================================================")

# -------------------- 导入第三方包 --------------------

import matplotlib.pyplot as plt  # 绘图

plt.switch_backend('agg')  # 使用非交互式后端
import os  # 文件与目录操作
from tqdm import tqdm  # 显示进度条

# -------------------- 导入项目内部模块 --------------------
from opts import parse_opts  # 解析命令行参数的工具函数

args = parse_opts()  # 解析所有命令行参数
wd = args.root_path  # 获取项目的根路径
os.chdir(wd)  # 切换到根路径
from utils.utils import *  # 导入项目常用工具函数
from models import *  # 导入项目中定义的各种模型
from opts import *
#######################################################################################
'''                                Output folders                                  '''
#######################################################################################
# -------------------- 创建输出文件夹 --------------------
path_to_output = os.path.join(wd,
                              'output/CcGAN_{}_{}_si{:.3f}_ka{:.3f}_{}_nDs{}_nDa{}_nGa{}_Dbs{}_Gbs{}'.format(
                                      args.GAN_arch, args.threshold_type, args.kernel_sigma,
                                      args.kappa, args.loss_type_gan,
                                      args.num_D_steps, args.num_grad_acc_d, args.num_grad_acc_g,
                                      args.batch_size_disc, args.batch_size_gene))
os.makedirs(path_to_output, exist_ok=True)
save_models_folder = os.path.join(path_to_output, 'saved_models')
os.makedirs(save_models_folder, exist_ok=True)
save_images_folder = os.path.join(path_to_output, 'saved_images')
os.makedirs(save_images_folder, exist_ok=True)
path_to_embed_models = os.path.join(wd, 'output/embed_models')
os.makedirs(path_to_embed_models, exist_ok=True)

#######################################################################################
'''                                  Evaluation                                     '''
#######################################################################################
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

def fn_sampleGAN_given_labels(labels, batch_size, to_numpy=True, denorm=True, verbose=False):
    """
    根据给定标签采样生成器生成的假图像

    参数:
        labels (np.ndarray): 归一化后的标签
        batch_size (int): 采样时每个批次的图像数
        to_numpy (bool, optional): 是否将输出转换为 numpy 数组，默认 True
        denorm (bool, optional): 是否对图像进行反归一化操作（映射回 [0,255]），默认 True
        verbose (bool, optional): 是否显示进度条，默认 False

    返回:
        tuple: (fake_images, fake_labels)
            fake_images: 生成的假图像集合
            fake_labels: 生成对应的标签
    """
    fake_images, fake_labels = sample_ccgan_given_labels(netG, net_y2h, labels,
                                                         batch_size=batch_size,
                                                         to_numpy=to_numpy, denorm=denorm,
                                                         verbose=verbose)
    return fake_images, fake_labels

if args.comp_FID:
    # -------------------- 加载用于评估的预训练模型 --------------------
    # 用于 FID 计算的编码器
    PreNetFID = encoder(dim_bottleneck=512).to(device)
    PreNetFID = nn.DataParallel(PreNetFID)
    Filename_PreCNNForEvalGANs = os.path.join(args.eval_ckpt_path, 'ckpt_AE_epoch_200_seed_2020_CVMode_False.pth')
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs, weights_only=True)
    PreNetFID.load_state_dict(checkpoint_PreNet['net_encoder_state_dict'])

    # 用于多样性评价（预测种族）的分类器
    PreNetDiversity = ResNet34_class_eval(num_classes=5, ngpu=torch.cuda.device_count()).to(device)  # 5 个种族
    Filename_PreCNNForEvalGANs_Diversity = os.path.join(args.eval_ckpt_path,
                                                         'ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_200_seed_2020_classify_5_races_CVMode_False.pth')
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_Diversity, weights_only=True)
    PreNetDiversity.load_state_dict(checkpoint_PreNet['net_state_dict'])

    # 用于计算 Label Score 的回归网络
    PreNetLS = ResNet34_regre_eval(ngpu=torch.cuda.device_count()).to(device)
    Filename_PreCNNForEvalGANs_LS = os.path.join(args.eval_ckpt_path,
                                                  'ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth')
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_LS, weights_only=True)
    PreNetLS.load_state_dict(checkpoint_PreNet['net_state_dict'])

    #####################
    # 生成每个标签下指定数量的假图像
    print("Start sampling {} fake images per label from GAN >>>".format(args.nfake_per_label))
    eval_labels = np.arange(1, args.max_label + 1)  # 原始标签
    eval_labels_norm = fn_norm_labels(eval_labels)   # 归一化标签
    num_eval_labels = len(eval_labels_norm)

    for i in tqdm(range(num_eval_labels)):
        label_i = eval_labels_norm[i]
        curr_fake_images, curr_fake_labels = fn_sampleGAN_given_labels(label_i * np.ones([args.nfake_per_label, 1]),
                                                                        args.samp_batch_size)
        if i == 0:
            fake_images = curr_fake_images
            fake_labels_assigned = curr_fake_labels.reshape(-1)
        else:
            fake_images = np.concatenate((fake_images, curr_fake_images), axis=0)
            fake_labels_assigned = np.concatenate((fake_labels_assigned, curr_fake_labels.reshape(-1)))
    assert len(fake_images) == args.nfake_per_label * num_eval_labels
    assert len(fake_labels_assigned) == args.nfake_per_label * num_eval_labels

    ## -------------------- 如果需要，将假图像导出以便 NIQE 评估 --------------------
    if args.dump_fake_for_NIQE:
        print("\n Dumping fake images for NIQE...")
        if args.niqe_dump_path == "None":
            dump_fake_images_folder = os.path.join(save_images_folder, 'fake_images')
        else:
            dump_fake_images_folder = os.path.join(args.niqe_dump_path, 'fake_images')
        os.makedirs(dump_fake_images_folder, exist_ok=True)
        for i in tqdm(range(len(fake_images))):
            label_i = int(fake_labels_assigned[i] * args.max_label)
            filename_i = os.path.join(dump_fake_images_folder, "{}_{}.png".format(i, label_i))
            os.makedirs(os.path.dirname(filename_i), exist_ok=True)
            image_i = fake_images[i]
            image_i_pil = Image.fromarray(image_i.transpose(1, 2, 0))
            image_i_pil.save(filename_i)
        sys.exit()

    print("End sampling!")
    print("\n We got {} fake images.".format(len(fake_images)))

    #####################
    # 评估：计算 FID、Label Score 等指标
    real_labels = raw_labels / args.max_label
    nfake_all = len(fake_images)
    nreal_all = len(raw_images)
    real_images = raw_images

    if args.comp_IS_and_FID_only:
        # -------------------- 计算整体 FID 和 IS --------------------
        indx_shuffle_real = np.arange(nreal_all); np.random.shuffle(indx_shuffle_real)
        indx_shuffle_fake = np.arange(nfake_all); np.random.shuffle(indx_shuffle_fake)
        FID = cal_FID(PreNetFID, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake],
                      batch_size=200, resize=None, norm_img=True)
        print("\n {}: FID of {} fake images: {:.3f}.".format(args.GAN_arch, nfake_all, FID))
        IS, IS_std = inception_score(imgs=fake_images[indx_shuffle_fake], num_classes=5, net=PreNetDiversity,
                                     cuda=True, batch_size=200, splits=10, normalize_img=True)
        print("\n {}: IS of {} fake images: {:.3f}({:.3f}).".format(args.GAN_arch, nfake_all, IS, IS_std))
    else:
        # -------------------- 分滑动窗口评估 FID、LS、Entropy 等指标 --------------------
        center_start = 1 + args.FID_radius
        center_stop = args.max_label - args.FID_radius
        centers_loc = np.arange(center_start, center_stop + 1)
        FID_over_centers = np.zeros(len(centers_loc))
        entropies_over_centers = np.zeros(len(centers_loc))
        labelscores_over_centers = np.zeros(len(centers_loc))
        num_realimgs_over_centers = np.zeros(len(centers_loc))
        for i in range(len(centers_loc)):
            center = centers_loc[i]
            interval_start = (center - args.FID_radius) / args.max_label
            interval_stop = (center + args.FID_radius) / args.max_label
            indx_real = np.where((real_labels >= interval_start) * (real_labels <= interval_stop) == True)[0]
            np.random.shuffle(indx_real)
            real_images_curr = real_images[indx_real]
            real_images_curr = (real_images_curr / 255.0 - 0.5) / 0.5
            num_realimgs_over_centers[i] = len(real_images_curr)
            indx_fake = np.where((fake_labels_assigned >= interval_start) * (fake_labels_assigned <= interval_stop) == True)[0]
            np.random.shuffle(indx_fake)
            fake_images_curr = fake_images[indx_fake]
            fake_images_curr = (fake_images_curr / 255.0 - 0.5) / 0.5
            fake_labels_assigned_curr = fake_labels_assigned[indx_fake]
            FID_over_centers[i] = cal_FID(PreNetFID, real_images_curr, fake_images_curr, batch_size=200, resize=None)
            predicted_class_labels = predict_class_labels(PreNetDiversity, fake_images_curr, batch_size=200, num_workers=args.num_workers)
            entropies_over_centers[i] = compute_entropy(predicted_class_labels)
            labelscores_over_centers[i], _ = cal_labelscore(PreNetLS, fake_images_curr, fake_labels_assigned_curr,
                                                            min_label_before_shift=0, max_label_after_shift=args.max_label,
                                                            batch_size=200, resize=None, num_workers=args.num_workers)
            print("\r Center:{}; Real:{}; Fake:{}; FID:{:.3f}; LS:{:.3f}; ET:{:.3f}.".format(
                center, len(real_images_curr), len(fake_images_curr),
                FID_over_centers[i], labelscores_over_centers[i], entropies_over_centers[i]))
        print("\n {} SFID: {:.3f}({:.3f}); min/max: {:.3f}/{:.3f}.".format(args.GAN_arch,
              np.mean(FID_over_centers), np.std(FID_over_centers), np.min(FID_over_centers), np.max(FID_over_centers)))
        print("\n {} LS over centers: {:.3f}({:.3f}); min/max: {:.3f}/{:.3f}.".format(args.GAN_arch,
              np.mean(labelscores_over_centers), np.std(labelscores_over_centers), np.min(labelscores_over_centers), np.max(labelscores_over_centers)))
        print("\n {} entropy over centers: {:.3f}({:.3f}); min/max: {:.3f}/{:.3f}.".format(args.GAN_arch,
              np.mean(entropies_over_centers), np.std(entropies_over_centers), np.min(entropies_over_centers), np.max(entropies_over_centers)))
        dump_fid_ls_entropy_over_centers_filename = os.path.join(path_to_output, 'fid_ls_entropy_over_centers')
        np.savez(dump_fid_ls_entropy_over_centers_filename, fids=FID_over_centers, labelscores=labelscores_over_centers,
                 entropies=entropies_over_centers, nrealimgs=num_realimgs_over_centers, centers=centers_loc)
        indx_shuffle_real = np.arange(nreal_all); np.random.shuffle(indx_shuffle_real)
        indx_shuffle_fake = np.arange(nfake_all); np.random.shuffle(indx_shuffle_fake)
        FID = cal_FID(PreNetFID, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake],
                      batch_size=200, resize=None, norm_img=True)
        print("\n {}: FID of {} fake images: {:.3f}.".format(args.GAN_arch, nfake_all, FID))
        ls_mean_overall, ls_std_overall = cal_labelscore(PreNetLS, fake_images, fake_labels_assigned,
                                                         min_label_before_shift=0, max_label_after_shift=args.max_label,
                                                         batch_size=200, resize=None, norm_img=True, num_workers=args.num_workers)
        print("\n {}: overall LS of {} fake images: {:.3f}({:.3f}).".format(args.GAN_arch, nfake_all, ls_mean_overall, ls_std_overall))
        eval_results_logging_fullpath = os.path.join(path_to_output, 'eval_results_{}.txt'.format(args.GAN_arch))
        if not os.path.isfile(eval_results_logging_fullpath):
            eval_results_logging_file = open(eval_results_logging_fullpath, "w")
            eval_results_logging_file.close()
        with open(eval_results_logging_fullpath, 'a') as eval_results_logging_file:
            eval_results_logging_file.write("\n===================================================================================================")
            eval_results_logging_file.write("\n Radius: {}.  \n".format(args.FID_radius))
            print(args, file=eval_results_logging_file)
            eval_results_logging_file.write("\n SFID: {:.3f} ({:.3f}).".format(np.mean(FID_over_centers), np.std(FID_over_centers)))
            eval_results_logging_file.write("\n LS: {:.3f} ({:.3f}).".format(ls_mean_overall, ls_std_overall))
            eval_results_logging_file.write("\n Diversity: {:.3f} ({:.3f}).".format(np.mean(entropies_over_centers), np.std(entropies_over_centers)))
            eval_results_logging_file.write("\n FID: {:.3f}.".format(FID))
    print("\n===================================================================================================")