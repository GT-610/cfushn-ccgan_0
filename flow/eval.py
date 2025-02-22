import os

import torch.nn as nn
from PIL import Image
from tqdm import tqdm  # 显示进度条

from config import cfg
from models.eval.ResNet_class_eval import ResNet34_class_eval
from models.eval.ResNet_regre_eval import ResNet34_regre_eval
from models.eval.autoencoder import encoder
from utils.eval_metrics import cal_FID, inception_score, cal_labelscore
from utils.utils import *  # 项目中常用工具函数

device = cfg.device


def sample(netG, net_y2h, cont_labels, class_labels, batch_size=500,
        to_numpy=True, denorm=True, verbose=True):
    n_fake = len(cont_labels)
    if batch_size > n_fake:
        batch_size = n_fake

    fake_images = []
    # 为了方便循环，将条件数组扩展以防止最后一个批次不足
    fake_cont_labels = np.concatenate((cont_labels, cont_labels[0:batch_size]), axis=0)
    fake_class_labels = np.concatenate((class_labels, class_labels[0:batch_size]), axis=0)

    netG = netG.to(device)
    netG.eval()
    net_y2h = net_y2h.to(device)
    net_y2h.eval()

    with torch.no_grad():
        if verbose:
            pb = SimpleProgressBar()
        n_img_got = 0
        while n_img_got < n_fake:
            z = torch.randn(batch_size, cfg.dim_gan, dtype=torch.float).to(device)
            # 获取当前批次的连续标签
            y_cont = torch.from_numpy(fake_cont_labels[n_img_got:(n_img_got + batch_size)]).type(
                    torch.float).view(-1, 1).to(device)
            # 获取当前批次的离散标签，转换为 long 型并展平
            y_class = torch.from_numpy(fake_class_labels[n_img_got:(n_img_got + batch_size)]).type(
                    torch.long).view(-1).to(device)
            # 通过联合标签映射网络生成条件嵌入
            cond = net_y2h(y_cont, y_class)
            # 生成假图像
            batch_fake_images = netG(z, cond)
            if denorm:
                # 将生成图像从 [-1,1] 映射到 [0,255]
                assert batch_fake_images.max().item() <= 1.0 and batch_fake_images.min().item() >= -1.0
                batch_fake_images = batch_fake_images * 0.5 + 0.5
                batch_fake_images = batch_fake_images * 255.0
                batch_fake_images = batch_fake_images.type(torch.uint8)
            fake_images.append(batch_fake_images.cpu())
            n_img_got += batch_size
            if verbose:
                pb.update(min(float(n_img_got) / n_fake, 1) * 100)
    fake_images = torch.cat(fake_images, dim=0)
    fake_images = fake_images[0:n_fake]  # 截取前 n_fake 张图像
    # 这里我们返回连续标签作为 fake_labels（或者你也可以返回联合标签），便于后续分析
    fake_labels = fake_cont_labels[0:n_fake]
    if to_numpy:
        fake_images = fake_images.numpy()
    else:
        fake_labels = torch.from_numpy(fake_labels)
    return fake_images, fake_labels


def evaluate_process(origin_data, netG, net_y2h):
    if cfg.comp_fid:
        raw_images, raw_cont_labels, _ = origin_data
        # -------------------- 加载用于评估的预训练模型 --------------------
        # 用于 FID 计算的编码器
        PreNetFID = encoder(dim_bottleneck=512).to(device)
        if cfg.gpu_parallel:
            PreNetFID = nn.DataParallel(PreNetFID)
        Filename_PreCNNForEvalGANs = os.path.join(cfg.eval_path,
                                                  'ckpt_AE_epoch_200_seed_2020_CVMode_False.pth')
        checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs, weights_only=True)
        PreNetFID.load_state_dict(checkpoint_PreNet['net_encoder_state_dict'])

        # 用于多样性评价（预测种族）的分类器
        PreNetDiversity = ResNet34_class_eval(num_classes=cfg.num_classes,
                                              ngpu=torch.cuda.device_count()).to(device)
        Filename_PreCNNForEvalGANs_Diversity = os.path.join(cfg.eval_path,
                                                            'ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_200_seed_2020_classify_5_races_CVMode_False.pth')
        checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_Diversity, weights_only=True)
        PreNetDiversity.load_state_dict(checkpoint_PreNet['net_state_dict'])

        # 用于计算 Label Score 的回归网络
        PreNetLS = ResNet34_regre_eval(ngpu=torch.cuda.device_count()).to(device)
        Filename_PreCNNForEvalGANs_LS = os.path.join(cfg.eval_path,
                                                     'ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth')
        checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_LS, weights_only=True)
        PreNetLS.load_state_dict(checkpoint_PreNet['net_state_dict'])

        #####################
        # 生成每个连续标签下指定数量的假图像
        print("Start sampling {} fake images per label from GAN >>>".format(cfg.n_fake_per_label))
        # 构造评价条件。这里假设你想针对每个连续标签生成假图像，
        # 同时采用固定离散标签条件，例如选择类别 2。你也可以按需要设置为其他策略。
        eval_cont_labels = np.arange(1, cfg.max_label + 1)  # 原始连续标签（例如年龄）
        eval_cont_labels_norm = fn_norm_labels(eval_cont_labels, cfg.max_label)  # 归一化到 [0,1]
        fixed_class = 2  # 固定的离散标签条件（例如类别2），请根据实际情况调整

        num_eval_labels = len(eval_cont_labels_norm)

        for i in tqdm(range(num_eval_labels)):
            label_i = eval_cont_labels_norm[i]
            # 为每个连续标签构造批量样本，连续标签数组形状 (nfake_per_label, 1)
            curr_cont = label_i * np.ones([cfg.n_fake_per_label, 1])
            # 同时构造对应的离散标签数组，全部为 fixed_class
            curr_class = fixed_class * np.ones([cfg.n_fake_per_label, 1])
            curr_fake_images, curr_fake_labels = sample(netG, net_y2h, curr_cont,
                                                        curr_class,
                                                        cfg.samp_batch_size)
            if i == 0:
                fake_images = curr_fake_images
                fake_labels_assigned = curr_fake_labels.reshape(-1)
            else:
                fake_images = np.concatenate((fake_images, curr_fake_images), axis=0)
                fake_labels_assigned = np.concatenate(
                        (fake_labels_assigned, curr_fake_labels.reshape(-1)))
        assert len(fake_images) == cfg.n_fake_per_label * num_eval_labels
        assert len(fake_labels_assigned) == cfg.n_fake_per_label * num_eval_labels

        ## -------------------- 如果需要，将假图像导出以便 NIQE 评估 --------------------
        if cfg.dump_fake_for_niqe:
            print("\n Dumping fake images for NIQE...")
            if cfg.niqe_dump_path == "None":
                dump_fake_images_folder = os.path.join(cfg.gan_output_path, 'saved_images',
                                                       'fake_images')
            else:
                dump_fake_images_folder = os.path.join(cfg.niqe_dump_path, 'fake_images')
            os.makedirs(dump_fake_images_folder, exist_ok=True)
            for i in tqdm(range(len(fake_images))):
                # 这里将连续标签反归一化（乘以 max_label）
                label_i = int(fake_labels_assigned[i] * cfg.max_label)
                filename_i = os.path.join(dump_fake_images_folder, "{}_{}.png".format(i, label_i))
                os.makedirs(os.path.dirname(filename_i), exist_ok=True)
                image_i = fake_images[i]
                # image_i = ((image_i*0.5+0.5)*255.0).astype(np.uint8)  # 此处已在 sample() 内完成 denorm
                image_i_pil = Image.fromarray(image_i.transpose(1, 2, 0))
                image_i_pil.save(filename_i)
            sys.exit()

        print("End sampling!")
        print("\n We got {} fake images.".format(len(fake_images)))

        #####################
        # 评估：计算 FID、Label Score、以及类别多样性指标（Entropy）
        real_labels = raw_cont_labels / cfg.max_label  # 对连续标签归一化
        nfake_all = len(fake_images)
        nreal_all = len(raw_images)
        real_images = raw_images

        if cfg.comp_is_and_fid_only:
            #####################
            # 计算整体 FID 和 IS
            indx_shuffle_real = np.arange(nreal_all)
            np.random.shuffle(indx_shuffle_real)
            indx_shuffle_fake = np.arange(nfake_all)
            np.random.shuffle(indx_shuffle_fake)
            FID = cal_FID(PreNetFID, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake],
                          batch_size=200, resize=None, norm_img=True)
            print("\n {}: FID of {} fake images: {:.3f}.".format(cfg.gan_arch, nfake_all, FID))
            IS, IS_std = inception_score(imgs=fake_images[indx_shuffle_fake], num_classes=5,
                                         net=PreNetDiversity,
                                         cuda=True, batch_size=200, splits=10, normalize_img=True)
            print(
                    "\n {}: IS of {} fake images: {:.3f}({:.3f}).".format(
                            cfg.gan_arch, nfake_all, IS, IS_std))
        else:
            #####################
            # 分滑动窗口评估 FID、Label Score 和类别多样性（Entropy）
            center_start = 1 + cfg.fid_radius
            center_stop = cfg.max_label - cfg.fid_radius
            centers_loc = np.arange(center_start, center_stop + 1)
            FID_over_centers = np.zeros(len(centers_loc))
            entropies_over_centers = np.zeros(len(centers_loc))
            labelscores_over_centers = np.zeros(len(centers_loc))
            num_realimgs_over_centers = np.zeros(len(centers_loc))
            for i in range(len(centers_loc)):
                center = centers_loc[i]
                interval_start = (center - cfg.fid_radius) / cfg.max_label
                interval_stop = (center + cfg.fid_radius) / cfg.max_label
                indx_real = \
                    np.where(
                            (real_labels >= interval_start) * (
                                    real_labels <= interval_stop) == True)[
                        0]
                np.random.shuffle(indx_real)
                real_images_curr = real_images[indx_real]
                real_images_curr = (real_images_curr / 255.0 - 0.5) / 0.5
                num_realimgs_over_centers[i] = len(real_images_curr)
                indx_fake = np.where((fake_labels_assigned >= interval_start) * (
                        fake_labels_assigned <= interval_stop) == True)[0]
                np.random.shuffle(indx_fake)
                fake_images_curr = fake_images[indx_fake]
                fake_images_curr = (fake_images_curr / 255.0 - 0.5) / 0.5
                fake_labels_assigned_curr = fake_labels_assigned[indx_fake]
                FID_over_centers[i] = cal_FID(PreNetFID, real_images_curr, fake_images_curr,
                                              batch_size=200, resize=None)
                predicted_class_labels = predict_class_labels(PreNetDiversity, fake_images_curr,
                                                              batch_size=200,
                                                              num_workers=cfg.num_workers)
                entropies_over_centers[i] = compute_entropy(predicted_class_labels)
                labelscores_over_centers[i], _ = cal_labelscore(PreNetLS, fake_images_curr,
                                                                fake_labels_assigned_curr,
                                                                min_label_before_shift=0,
                                                                max_label_after_shift=cfg.max_label,
                                                                batch_size=200, resize=None,
                                                                num_workers=cfg.num_workers)
                print("\r Center:{}; Real:{}; Fake:{}; FID:{:.3f}; LS:{:.3f}; ET:{:.3f}.".format(
                        center, len(real_images_curr), len(fake_images_curr),
                        FID_over_centers[i], labelscores_over_centers[i],
                        entropies_over_centers[i]))
            print("\n {} SFID: {:.3f}({:.3f}); min/max: {:.3f}/{:.3f}.".format(
                    cfg.gan_arch, np.mean(FID_over_centers),
                    np.std(FID_over_centers),
                    np.min(FID_over_centers),
                    np.max(FID_over_centers)))
            print("\n {} LS over centers: {:.3f}({:.3f}); min/max: {:.3f}/{:.3f}.".format(
                    cfg.gan_arch,
                    np.mean(labelscores_over_centers),
                    np.std(labelscores_over_centers),
                    np.min(labelscores_over_centers),
                    np.max(labelscores_over_centers)))
            print("\n {} entropy over centers: {:.3f}({:.3f}); min/max: {:.3f}/{:.3f}.".format(
                    cfg.gan_arch,
                    np.mean(entropies_over_centers), np.std(entropies_over_centers),
                    np.min(entropies_over_centers), np.max(entropies_over_centers)))
            dump_fid_ls_entropy_over_centers_filename = os.path.join(cfg.gan_output_path,
                                                                     'fid_ls_entropy_over_centers')
            np.savez(dump_fid_ls_entropy_over_centers_filename, fids=FID_over_centers,
                     labelscores=labelscores_over_centers,
                     entropies=entropies_over_centers, nrealimgs=num_realimgs_over_centers,
                     centers=centers_loc)
            indx_shuffle_real = np.arange(nreal_all)
            np.random.shuffle(indx_shuffle_real)
            indx_shuffle_fake = np.arange(nfake_all)
            np.random.shuffle(indx_shuffle_fake)
            FID = cal_FID(PreNetFID, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake],
                          batch_size=200, resize=None, norm_img=True)
            print("\n {}: FID of {} fake images: {:.3f}.".format(cfg.gan_arch, nfake_all, FID))
            ls_mean_overall, ls_std_overall = cal_labelscore(PreNetLS, fake_images,
                                                             fake_labels_assigned,
                                                             min_label_before_shift=0,
                                                             max_label_after_shift=cfg.max_label,
                                                             batch_size=200, resize=None,
                                                             norm_img=True,
                                                             num_workers=cfg.num_workers)
            print(
                    "\n {}: overall LS of {} fake images: {:.3f}({:.3f}).".format(cfg.gan_arch,
                                                                                  nfake_all,
                                                                                  ls_mean_overall,
                                                                                  ls_std_overall))
            eval_results_logging_fullpath = os.path.join(
                    cfg.gan_output_path, 'eval_results_{}.txt'.format(cfg.gan_arch))
            if not os.path.isfile(eval_results_logging_fullpath):
                with open(eval_results_logging_fullpath, "w") as eval_results_logging_file:
                    eval_results_logging_file.write("")
            with open(eval_results_logging_fullpath, 'a') as eval_results_logging_file:
                eval_results_logging_file.write(
                        "\n===================================================================================================")
                eval_results_logging_file.write("\n Radius: {}.  \n".format(cfg.fid_radius))
                # print(args, file=eval_results_logging_file)
                eval_results_logging_file.write(
                        "\n SFID: {:.3f} ({:.3f}).".format(np.mean(FID_over_centers),
                                                           np.std(FID_over_centers)))
                eval_results_logging_file.write(
                        "\n LS: {:.3f} ({:.3f}).".format(ls_mean_overall, ls_std_overall))
                eval_results_logging_file.write(
                        "\n Diversity: {:.3f} ({:.3f}).".format(np.mean(entropies_over_centers),
                                                                np.std(entropies_over_centers)))
                eval_results_logging_file.write("\n FID: {:.3f}.".format(FID))
        print(
                "\n===================================================================================================")
