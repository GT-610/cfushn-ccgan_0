import os

import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from config import cfg
from models.eval.ResNet_class_eval import ResNet34_class_eval
from models.eval.ResNet_regre_eval import ResNet34_regre_eval
from models.eval.autoencoder import encoder
from utils.eval_metrics import cal_FID, inception_score, cal_labelscore
from utils.utils import *

device = cfg.device
np_rng = np.random.default_rng(cfg.seed)


def sample(netG, net_y2h, cont_labels, class_labels, batch_size=500,
        to_numpy=True, denorm=True, verbose=True):
    total = len(cont_labels)
    if batch_size > total:
        batch_size = total

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
        while n_img_got < total:
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
                pb.update(min(float(n_img_got) / total, 1) * 100)
    fake_images = torch.cat(fake_images, dim=0)
    fake_images = fake_images[0:total]  # 截取前 n_fake 张图像
    # 这里我们返回连续标签作为 fake_labels（或者你也可以返回联合标签），便于后续分析
    fake_labels = fake_cont_labels[0:total]
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
        Filename_PreCNNForEvalGANs = os.path.join(cfg.eval_path, str(cfg.pretrained_ae_pth))
        checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs, weights_only=True)
        PreNetFID.load_state_dict(checkpoint_PreNet['net_encoder_state_dict'])

        # 用于多样性评价（预测种族）的分类器
        PreNetDiversity = ResNet34_class_eval(num_classes=cfg.num_classes,
                                              ngpu=torch.cuda.device_count()).to(device)
        Filename_PreCNNForEvalGANs_Diversity = os.path.join(cfg.eval_path,
                                                            str(cfg.pretrained_cnn4class_pth))
        checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_Diversity, weights_only=True)
        PreNetDiversity.load_state_dict(checkpoint_PreNet['net_state_dict'])

        # 用于计算 Label Score 的回归网络
        PreNetLS = ResNet34_regre_eval(ngpu=torch.cuda.device_count()).to(device)
        Filename_PreCNNForEvalGANs_LS = os.path.join(cfg.eval_path,
                                                     str(cfg.pretrained_cnn4cont_pth))
        checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_LS, weights_only=True)
        PreNetLS.load_state_dict(checkpoint_PreNet['net_state_dict'])

        # -------------------- 生成每个连续标签下指定数量的假图像 --------------------
        print(f"Start sampling {cfg.n_fake_per_label} fake images per label from GAN >>>")
        # 构造评价条件。这里假设你想针对每个连续标签生成假图像，
        # 同时采用固定离散标签条件，例如选择类别 2。你也可以按需要设置为其他策略。
        eval_cont_labels = np.arange(cfg.min_label, cfg.max_label + 1)  # 原始连续标签（例如年龄）
        eval_cont_labels_norm = fn_norm_labels(eval_cont_labels, cfg.max_label)  # 归一化到 [0,1]

        num_eval_labels = len(eval_cont_labels_norm)

        fake_images = None
        fake_labels_assigned = None
        for i in tqdm(range(num_eval_labels)):
            label_i = eval_cont_labels_norm[i]
            # 为每个连续标签构造批量样本，连续标签数组形状 (n_fake_per_label, 1)
            curr_cont = label_i * np.ones([cfg.n_fake_per_label, 1])
            # 同时构造对应的离散标签数组(随机类别下标)
            curr_class = np_rng.integers(cfg.num_classes, size=(cfg.n_fake_per_label, 1))
            curr_fake_images, curr_fake_labels = sample(netG, net_y2h, curr_cont, curr_class,
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
            dump_fake_images_folder = cfg.niqe_dump_path if cfg.niqe_dump_path != "None" \
                else os.path.join(cfg.gan_output_path, 'saved_images', 'fake_images')
            os.makedirs(dump_fake_images_folder, exist_ok=True)
            for i in tqdm(range(len(fake_images))):
                # 这里将连续标签反归一化（乘以 max_label）
                label_i = int(fake_labels_assigned[i] * cfg.max_label)
                filename_i = os.path.join(dump_fake_images_folder, f"{i}_{label_i}.png")
                os.makedirs(os.path.dirname(filename_i), exist_ok=True)
                image_i = fake_images[i]
                # image_i = ((image_i*0.5+0.5)*255.0).astype(np.uint8)  # 此处已在 sample() 内完成 denorm
                image_i_pil = Image.fromarray(image_i.transpose(1, 2, 0))
                image_i_pil.save(filename_i)
            sys.exit()

        print("End sampling!")
        print(f"\n We got {len(fake_images)} fake images.")

        #####################
        # 评估：计算 FID、Label Score、以及类别多样性指标（Entropy）
        real_labels = raw_cont_labels / cfg.max_label  # 对连续标签归一化
        n_fake_all = len(fake_images)
        n_real_all = len(raw_images)
        real_images = raw_images

        if cfg.comp_is_and_fid_only:
            #####################
            # 计算整体 FID 和 IS
            index_shuffle_real = np.arange(n_real_all)
            np_rng.shuffle(index_shuffle_real)
            index_shuffle_fake = np.arange(n_fake_all)
            np_rng.shuffle(index_shuffle_fake)
            FID = cal_FID(PreNetFID, real_images[index_shuffle_real],
                          fake_images[index_shuffle_fake],
                          batch_size=200, resize=None, norm_img=True)
            print(f"\n {cfg.gan_arch}: FID of {n_fake_all} fake images: {FID:.3f}.")
            IS, IS_std = inception_score(imgs=fake_images[index_shuffle_fake], num_classes=5,
                                         net=PreNetDiversity,
                                         cuda=True, batch_size=200, splits=10, normalize_img=True)
            print(f"\n {cfg.gan_arch}: IS of {n_fake_all} fake images: {IS:.3f}({IS_std:.3f}).")
        else:
            #####################
            # 分滑动窗口评估 FID、Label Score 和类别多样性（Entropy）
            center_start = 1 + cfg.fid_radius
            center_stop = cfg.max_label - cfg.fid_radius
            centers_loc = np.arange(center_start, center_stop + 1)
            FID_over_centers = np.zeros(len(centers_loc))
            entropies_over_centers = np.zeros(len(centers_loc))
            label_scores_over_centers = np.zeros(len(centers_loc))
            num_real_imgs_over_centers = np.zeros(len(centers_loc))
            for i in range(len(centers_loc)):
                center = centers_loc[i]
                interval_start = (center - cfg.fid_radius) / cfg.max_label
                interval_stop = (center + cfg.fid_radius) / cfg.max_label
                index_real = np.where(
                        (real_labels >= interval_start) * (real_labels <= interval_stop) == True)[0]
                np_rng.shuffle(index_real)
                real_images_curr = real_images[index_real]
                real_images_curr = (real_images_curr / 255.0 - 0.5) / 0.5
                num_real_imgs_over_centers[i] = len(real_images_curr)
                index_fake = np.where((fake_labels_assigned >= interval_start) * (
                        fake_labels_assigned <= interval_stop) == True)[0]
                np_rng.shuffle(index_fake)
                fake_images_curr = fake_images[index_fake]
                fake_images_curr = (fake_images_curr / 255.0 - 0.5) / 0.5
                fake_labels_assigned_curr = fake_labels_assigned[index_fake]
                FID_over_centers[i] = cal_FID(PreNetFID, real_images_curr, fake_images_curr,
                                              batch_size=200, resize=None)
                predicted_class_labels = predict_class_labels(PreNetDiversity, fake_images_curr,
                                                              batch_size=200,
                                                              num_workers=cfg.num_workers)
                entropies_over_centers[i] = compute_entropy(predicted_class_labels)
                label_scores_over_centers[i], _ = cal_labelscore(PreNetLS, fake_images_curr,
                                                                 fake_labels_assigned_curr,
                                                                 min_label_before_shift=0,
                                                                 max_label_after_shift=cfg.max_label,
                                                                 batch_size=200, resize=None,
                                                                 num_workers=cfg.num_workers)
                print("\r Center:{}; Real:{}; Fake:{}; FID:{:.3f}; LS:{:.3f}; ET:{:.3f}.".format(
                        center, len(real_images_curr), len(fake_images_curr),
                        FID_over_centers[i], label_scores_over_centers[i],
                        entropies_over_centers[i]))
            print("\n {} SFID: {:.3f}({:.3f}); min/max: {:.3f}/{:.3f}.".format(
                    cfg.gan_arch, np.mean(FID_over_centers),
                    np.std(FID_over_centers),
                    np.min(FID_over_centers),
                    np.max(FID_over_centers)))
            print("\n {} LS over centers: {:.3f}({:.3f}); min/max: {:.3f}/{:.3f}.".format(
                    cfg.gan_arch,
                    np.mean(label_scores_over_centers),
                    np.std(label_scores_over_centers),
                    np.min(label_scores_over_centers),
                    np.max(label_scores_over_centers)))
            print("\n {} entropy over centers: {:.3f}({:.3f}); min/max: {:.3f}/{:.3f}.".format(
                    cfg.gan_arch,
                    np.mean(entropies_over_centers), np.std(entropies_over_centers),
                    np.min(entropies_over_centers), np.max(entropies_over_centers)))
            dump_fid_ls_entropy_over_centers_filename = os.path.join(cfg.gan_output_path,
                                                                     'fid_ls_entropy_over_centers')
            np.savez(dump_fid_ls_entropy_over_centers_filename, fids=FID_over_centers,
                     labelscores=label_scores_over_centers,
                     entropies=entropies_over_centers, nrealimgs=num_real_imgs_over_centers,
                     centers=centers_loc)
            index_shuffle_real = np.arange(n_real_all)
            np_rng.shuffle(index_shuffle_real)
            index_shuffle_fake = np.arange(n_fake_all)
            np_rng.shuffle(index_shuffle_fake)
            FID = cal_FID(PreNetFID, real_images[index_shuffle_real],
                          fake_images[index_shuffle_fake],
                          batch_size=200, resize=None, norm_img=True)
            print(f"\n {cfg.gan_arch}: FID of {n_fake_all} fake images: {FID:.3f}.")
            ls_mean_overall, ls_std_overall = cal_labelscore(PreNetLS, fake_images,
                                                             fake_labels_assigned,
                                                             min_label_before_shift=0,
                                                             max_label_after_shift=cfg.max_label,
                                                             batch_size=200, resize=None,
                                                             norm_img=True,
                                                             num_workers=cfg.num_workers)
            print(f"\n {cfg.gan_arch}: overall LS of {n_fake_all} "
                  f"fake images: {ls_mean_overall:.3f}({ls_std_overall:.3f}).")
            eval_results_logging_fullpath = os.path.join(
                    cfg.gan_output_path, f'eval_results_{cfg.gan_arch}.txt')
            if not os.path.isfile(eval_results_logging_fullpath):
                with open(eval_results_logging_fullpath, "w") as eval_txt:
                    eval_txt.write("")
            with open(eval_results_logging_fullpath, 'a') as eval_txt:
                eval_txt.write("\n================================================================")
                eval_txt.write(f"\n Radius: {cfg.fid_radius}.  \n")
                eval_txt.write(
                        "\n SFID ↓: {:.3f} ({:.3f})."
                        .format(np.mean(FID_over_centers), np.std(FID_over_centers)))
                eval_txt.write(
                        "\n LS ↓: {:.3f} ({:.3f}).".format(ls_mean_overall, ls_std_overall))
                eval_txt.write(
                        "\n Diversity ↑: {:.3f} ({:.3f})."
                        .format(np.mean(entropies_over_centers), np.std(entropies_over_centers)))
                eval_txt.write("\n FID ↓: {:.3f}.".format(FID))
                eval_txt.write("\n================================================================")
                # 记录所有参数
                eval_txt.write("\n" + cfg.pretty_str())

        print("\n===============================================================================")
