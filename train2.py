import copy  # 深拷贝对象
import gc  # 垃圾回收工具

import h5py  # 读取 HDF5 文件
from tqdm import tqdm  # 显示进度条

from models.ResNet_embed2 import *
from models.sngan2 import *
from train_ccgan2 import train_ccgan  # 导入 GAN 训练及采样函数
from train_net_for_label_embed2 import *
from utils.ipc_util import register_signal_handler, get_s1, get_s2
from utils.log_util import cy_log
from utils.utils2 import *  # 导入项目常用工具函数

# --------------------------- 注册信号事件; 并定义trap,根据信号执行操作 ---------------------------
register_signal_handler()


def my_trap(signal_num: int):
    trigger = False
    if signal_num == 10:
        trigger = (get_s1() > 0)
    elif signal_num == 12:
        trigger = (get_s2() > 0)
    if trigger:
        cy_log("==============检测到退出信号!!!!!!!!!!!!==============")
        # save_checkpoint(unet, optimizer, epoch, epoch_loss)
        cy_log("==============临时保存完毕,退出训练进程==============")
        sys.exit(0)


# --------------------------- Data loader ---------------------------
# ------ 加载数据 ------
# 数据文件名：根据图像尺寸构造 h5 文件名（例如 UTKFace_64x64.h5）
data_filename = DATA_PATH + '/UTKFace_{}x{}.h5'.format(IMG_SIZE, IMG_SIZE)
hf = h5py.File(data_filename, 'r')

# 加载连续标签（例如年龄），并转为 float 类型
age_labels = hf['labels'][:]  # 注意：这里假设 h5 文件中 'labels' 存储年龄
age_labels = age_labels.astype(float)
# 加载离散标签（例如人种），并转为 int 类型
race_labels = hf['races'][:]  # 注意：这里假设 h5 文件中 'races' 存储人种类别，取值为 0~4（共 5 类）
race_labels = race_labels.astype(int)
# 加载图像数据
images = hf['images'][:]
hf.close()

# ------ 数据子集选择 ------
# 根据连续标签（年龄）的范围 [min_label, max_label] 选择子集
selected_age_labels = np.arange(MIN_LABEL, MAX_LABEL + 1)
images_subset = None
age_labels_subset = None
race_labels_subset = None
for i in range(len(selected_age_labels)):
    curr_age = selected_age_labels[i]
    # 找出年龄等于当前值的所有样本索引
    index_curr = np.where(age_labels == curr_age)[0]
    if i == 0:
        images_subset = images[index_curr]
        age_labels_subset = age_labels[index_curr]
        race_labels_subset = race_labels[index_curr]
    else:
        images_subset = np.concatenate((images_subset, images[index_curr]), axis=0)
        age_labels_subset = np.concatenate((age_labels_subset, age_labels[index_curr]))
        race_labels_subset = np.concatenate((race_labels_subset, race_labels[index_curr]))
# 更新数据集：只保留所选子集
images = images_subset
age_labels = age_labels_subset
race_labels = race_labels_subset
del images_subset, age_labels_subset, race_labels_subset
gc.collect()

# 保留数据的一个副本（原始数据）
raw_images = copy.deepcopy(images)
raw_age_labels = copy.deepcopy(age_labels)
raw_race_labels = copy.deepcopy(race_labels)

# ------ 每个连续标签（年龄）最多保留指定数量的图像 ------
image_num_threshold = MAX_NUM_IMG_PER_LABEL
print("\n Original set has {} images; For each continuous label, take no more than {} images>>>".format(
        len(images), image_num_threshold))
# 获取所有唯一的年龄值（连续标签）
unique_age_tmp = np.sort(np.array(list(set(age_labels))))
sel_index = None
for i in tqdm(range(len(unique_age_tmp))):
    index_i = np.where(age_labels == unique_age_tmp[i])[0]
    if len(index_i) > image_num_threshold:
        np.random.shuffle(index_i)
        index_i = index_i[0:image_num_threshold]
    if i == 0:
        sel_index = index_i
    else:
        sel_index = np.concatenate((sel_index, index_i))
    # 根据选取的索引更新图像、连续标签和离散标签
images = images[sel_index]
age_labels = age_labels[sel_index]
race_labels = race_labels[sel_index]
print("{} images left.".format(len(images)))

# ------ 复制少数样本以缓解连续标签（age）的不平衡 ------
# 这里仅针对连续标签（例如年龄）进行复制，不涉及离散标签（例如人种）
# 计算用于复制的最大图像数量：不能超过 max_num_img_per_label_after_replica 和 max_num_img_per_label 中的较小值
max_num_img_per_label_after_replica = np.min(
        [MAX_NUM_IMG_PER_LABEL_AFTER_REPLICA, MAX_NUM_IMG_PER_LABEL])
if max_num_img_per_label_after_replica > 1:
    # 获取所有唯一的连续标签值（例如年龄）
    unique_age = np.sort(np.array(list(set(age_labels))))
    num_labels_replicated = 0  # 用于统计实际执行复制的标签数量
    print("Start replicating minority samples for continuous labels >>>")
    images_replica = None  # 用于存放复制出来的图像
    age_labels_replica = None  # 用于存放复制出来的连续标签
    race_labels_replica = None  # 新增：用于存放复制出来的离散标签

    # 遍历每个唯一连续标签
    for i in tqdm(range(len(unique_age))):
        curr_age = unique_age[i]
        # 找出当前标签对应的所有样本索引
        index_i = np.where(age_labels == curr_age)[0]
        # 如果当前标签样本数量不足，则进行复制
        if len(index_i) < max_num_img_per_label_after_replica:
            num_img_less = max_num_img_per_label_after_replica - len(index_i)
            # 随机从当前样本中复制缺少的数量（允许重复）
            index_replica = np.random.choice(index_i, size=num_img_less, replace=True)
            if num_labels_replicated == 0:
                images_replica = images[index_replica]
                age_labels_replica = age_labels[index_replica]
                race_labels_replica = race_labels[index_replica]  # 同步复制离散标签
            else:
                images_replica = np.concatenate((images_replica, images[index_replica]), axis=0)
                age_labels_replica = np.concatenate((age_labels_replica, age_labels[index_replica]))
                race_labels_replica = np.concatenate(
                        (race_labels_replica, race_labels[index_replica]), axis=0)  # 同步复制
            num_labels_replicated += 1
    # 将复制的样本与原始数据合并
    images = np.concatenate((images, images_replica), axis=0)
    age_labels = np.concatenate((age_labels, age_labels_replica), axis=0)
    race_labels = np.concatenate((race_labels, race_labels_replica), axis=0)  # 同步合并
    # 注意：离散标签（race_labels）无需复制，因为类别样本通常较充足，
    # 但若需要平衡离散标签可额外增加相应复制逻辑
    print("We replicate {} images and continuous labels \n".format(len(images_replica)))
    del images_replica, age_labels_replica, race_labels_replica
    gc.collect()

# -------------------- 连续标签归一化 --------------------
print("\n Range of unnormalized continuous labels: ({},{})".format(np.min(age_labels),
                                                                   np.max(age_labels)))
# 使用辅助函数对连续标签归一化到 [0,1]（需要传入最大标签值 max_label）
age_labels = fn_norm_labels(age_labels, MAX_LABEL)
print("\n Range of normalized continuous labels: ({},{})".format(np.min(age_labels),
                                                                 np.max(age_labels)))
# 获取归一化后唯一的连续标签（用于后续分析或训练数据准备）
unique_age_norm = np.sort(np.array(list(set(age_labels))))
print("Unique race labels before adjustment:", np.unique(race_labels))

# -------------------- 根据数据统计自动计算 kernel_sigma 与 kappa --------------------
if KERNEL_SIGMA < 0:
    std_label = np.std(age_labels)
    kernel_sigma = 1.06 * std_label * (len(age_labels)) ** (-1 / 5)
    print("\n Use rule-of-thumb formula to compute kernel_sigma >>>")
    print("\n The std of {} age labels is {} so the kernel sigma is {}".format(
            len(age_labels), std_label, kernel_sigma))

if KAPPA < 0:
    n_unique = len(unique_age_norm)
    diff_list = []
    for i in range(1, n_unique):
        diff_list.append(unique_age_norm[i] - unique_age_norm[i - 1])
    kappa_base = np.abs(KAPPA) * np.max(np.array(diff_list))
    if THRESHOLD_TYPE == "hard":
        kappa = kappa_base
    else:
        kappa = 1 / kappa_base ** 2

#######################################################################################
'''               Pre-trained CNN and GAN for label embedding                       '''
#######################################################################################
# -------------------- 定义预训练模型的 checkpoint 文件名 --------------------
net_embed_filename_ckpt = os.path.join(path_to_embed_models,
                                       'ckpt_{}_epoch_{}_seed_{}_v2.pth'.format(NET_EMBED_TYPE,
                                                                                EPOCH_CNN_EMBED,
                                                                                SEED))
net_y2h_filename_ckpt = os.path.join(path_to_embed_models,
                                     'ckpt_net_y2h_epoch_{}_seed_{}_v2.pth'.format(EPOCH_NET_Y2H,
                                                                                   SEED))

print("\n " + net_embed_filename_ckpt)
print("\n " + net_y2h_filename_ckpt)

# -------------------- 构建训练集和 DataLoader --------------------
# 这里假设 ImgsDataset 类已经支持同时接受三个参数：图像、连续标签、离散标签，并自动归一化图像
trainset = ImgsDataset_v2(images, age_labels, race_labels, normalize=True)
trainloader_embed_net = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE_EMBED,
                                                    shuffle=True, num_workers=NUM_WORKERS)

# -------------------- 构建图像嵌入模型 net_embed --------------------
net_embed = None
if NET_EMBED_TYPE == "ResNet18_embed":
    net_embed = ResNet18_embed_v2(dim_embed=DIM_EMBED)
elif NET_EMBED_TYPE == "ResNet34_embed":
    net_embed = ResNet34_embed_v2(dim_embed=DIM_EMBED)
elif NET_EMBED_TYPE == "ResNet50_embed":
    net_embed = ResNet50_embed_v2(dim_embed=DIM_EMBED)
net_embed = net_embed.to(device)

# -------------------- 构建标签映射模型 net_y2h --------------------
net_y2h = model_y2h_v2(dim_embed=DIM_EMBED)
net_y2h = net_y2h.to(device)

## (1). 训练 net_embed：将图像映射到嵌入空间，然后通过 h2y 映射回标签（x2h+h2y）
if not os.path.isfile(net_embed_filename_ckpt):
    print("\n Start training CNN for label embedding >>>")
    net_embed = train_net_embed(net=net_embed, net_name=net_embed,
                                trainloader=trainloader_embed_net,
                                testloader=None, epochs=EPOCH_CNN_EMBED,
                                resume_epoch=RESUME_EPOCH_CNN_EMBED,
                                lr_base=BASE_LR_X2Y, lr_decay_factor=0.1, lr_decay_epochs=[80, 140],
                                weight_decay=1e-4, path_to_ckpt=path_to_embed_models)
    # 保存训练好的 net_embed 模型
    torch.save({
        'net_state_dict': net_embed.state_dict(),
    }, net_embed_filename_ckpt)
else:
    print("\n net_embed ckpt already exists")
    print("\n Loading...")
    checkpoint = torch.load(net_embed_filename_ckpt, weights_only=True, map_location=device)
    net_embed.load_state_dict(checkpoint['net_state_dict'])

## (2). 训练 net_y2h：将标签映射到与图像嵌入相同的空间
if not os.path.isfile(net_y2h_filename_ckpt):
    print("\n Start training net_y2h >>>")
    net_y2h = train_net_y2h(age_labels, race_labels, net_y2h, net_embed, epochs=EPOCH_NET_Y2H,
                            lr_base=BASE_LR_Y2H, lr_decay_factor=0.1,
                            lr_decay_epochs=[150, 250, 350],
                            weight_decay=1e-4, batch_size=128)
    # 保存训练好的 net_y2h 模型
    torch.save({
        'net_state_dict': net_y2h.state_dict(),
    }, net_y2h_filename_ckpt)
else:
    print("\n net_y2h ckpt already exists")
    print("\n Loading...")
    checkpoint = torch.load(net_y2h_filename_ckpt, weights_only=True, map_location=device)
    net_y2h.load_state_dict(checkpoint['net_state_dict'])

## -------------------- 简单测试：检查连续标签映射是否正确 --------------------
# 从连续标签中随机选择 10 个用于测试映射效果
index_tmp = np.arange(len(unique_age_norm))
np.random.shuffle(index_tmp)
index_tmp = index_tmp[:10]  # 60个唯一标签索引打乱后,取前10个
# labels_tmp: 形状 (10, 1)，注意这里连续标签已经归一化到 [0,1]
labels_tmp = unique_age_norm[index_tmp].reshape(-1, 1)
labels_tmp = torch.from_numpy(labels_tmp).type(torch.float).to(device)
# 添加噪声，模拟连续标签的不确定性
epsilons_tmp = np.random.normal(0, 0.2, len(labels_tmp))
epsilons_tmp = torch.from_numpy(epsilons_tmp).view(-1, 1).type(torch.float).to(device)
labels_noise_tmp = torch.clamp(labels_tmp + epsilons_tmp, 0.0, 1.0)

# 对于离散标签测试，我们可以选择一个固定的类别，例如 2（假设类别取值为 0~4）
fixed_class = torch.full((labels_tmp.size(0),), 2, dtype=torch.long).to(device)

net_embed.eval()
net_y2h.eval()

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
print("\n Continuous labels vs reconstructed labels:")
print("Original, Rec (no noise), Rec (with noise)")
print(results1)

# 计算连续标签误差（均方差），作为指标
labels_diff = (labels_tmp_np - labels_noise_np) ** 2
# 这里也可以计算隐层表示差异等
print("\n Continuous labels diff (squared):")
print(labels_diff)

# 输出离散标签部分的预测
# 对 logits 做 softmax 并取最大概率对应的类别
import torch.nn.functional as F

predicted_class = F.softmax(torch.tensor(class_logits), dim=1).argmax(dim=1).numpy()
print("\n Fixed discrete label: 2, Predicted discrete label from net_y2h fusion branch:")
print(predicted_class)

# 释放内存：将模型放回 CPU
net_embed = net_embed.cpu()
net_y2h = net_y2h.cpu()

#######################################################################################
'''                                    GAN training                                 '''
#######################################################################################
print("CcGAN: {}, {}, Sigma is {:.4f}, Kappa is {:.4f}.".format(GAN_ARCH, THRESHOLD_TYPE,
                                                                KERNEL_SIGMA, KAPPA))
save_images_in_train_folder = os.path.join(save_images_folder, 'images_in_train')
os.makedirs(save_images_in_train_folder, exist_ok=True)

start = timeit.default_timer()
print("\n Begin Training >>>")
ckpt_gan_path = os.path.join(save_models_folder, 'ckpt_niter_{}.pth'.format(N_ITERS))
print(ckpt_gan_path)
netG = None
netD = None
if not os.path.isfile(ckpt_gan_path):
    # 根据 GAN 架构选择生成器与判别器
    if GAN_ARCH == "SAGAN":
        # netG = sagan_generator(nz=dim_gan, dim_embed=dim_embed).to(device)
        # netD = sagan_discriminator(dim_embed=dim_embed).to(device)
        pass
    else:
        netG = sngan_generator(nz=DIM_GAN, dim_embed=DIM_EMBED).to(device)
        netD = sngan_discriminator(dim_embed=DIM_EMBED).to(device)
    netG = nn.DataParallel(netG)  # 使用多GPU并行训练
    netD = nn.DataParallel(netD)

    # 调用 train_ccgan 函数进行 GAN 训练
    netG, netD = train_ccgan(KERNEL_SIGMA, KAPPA, images, age_labels, race_labels, netG, netD,
                             net_y2h,
                             save_images_folder=save_images_in_train_folder,
                             save_models_folder=save_models_folder)
    # 保存训练好的生成器模型
    torch.save({'netG_state_dict': netG.state_dict()}, ckpt_gan_path)
else:
    print("Loading pre-trained generator >>>")
    checkpoint = torch.load(ckpt_gan_path, weights_only=True, map_location=device)
    # 根据 GAN 架构选择生成器
    if GAN_ARCH == "SAGAN":
        # netG = sagan_generator(nz=dim_gan, dim_embed=dim_embed).to(device)
        pass
    else:
        netG = sngan_generator(nz=DIM_GAN, dim_embed=DIM_EMBED).to(device)
    netG = nn.DataParallel(netG)
    netG.load_state_dict(checkpoint['netG_state_dict'])

stop = timeit.default_timer()
print("GAN training finished; Time elapses: {}s".format(stop - start))
