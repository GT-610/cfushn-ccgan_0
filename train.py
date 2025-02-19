import h5py  # 读取 HDF5 文件
from tqdm import tqdm  # 显示进度条

from config.config import *
from models.ResNet_embed import *
from models.sngan import *
from train_ccgan import train_ccgan
from train_net_for_label_embed import *
from utils.data_util import get_distribution_table
from utils.utils import *

#######################################################################################
'''                              数据预处理                                          '''
#######################################################################################
# --------------------------- 加载数据 ---------------------------
# 数据文件名：根据图像尺寸构造 h5 文件名（例如 UTKFace_64x64.h5）
data_filename = DATA_PATH + '/UTKFace_{}x{}.h5'.format(IMG_SIZE, IMG_SIZE)
hf = h5py.File(data_filename, 'r')

# 加载图像数据
images = hf['images'][:]
# 加载连续标签（例如年龄），并转为 float 类型
cont_labels = (hf['labels'][:]).astype(float)  # numpy.ndarray (存储年龄标签)
# 加载离散标签（例如人种），并转为 int 类型
class_labels = (hf['races'][:]).astype(int)  # numpy.ndarray (存储人种标签)
hf.close()

# --------------------------- 数据子集选择 scoping ---------------------------
# 根据连续标签（年龄）的范围 [min_label, max_label] 筛选数据
selected_cont_labels = np.arange(MIN_LABEL, MAX_LABEL + 1)
select_index_arr_arr = []  # [[],[],[],...]
for i in range(len(selected_cont_labels)):
    curr_cont = selected_cont_labels[i]
    # 找出年龄等于当前值的所有样本索引
    index_curr = np.where(cont_labels == curr_cont)[0]  # (用到广播机制)
    select_index_arr_arr.append(index_curr)
# 更新数据集：只保留所选子集
select_index_arr = np.concatenate(select_index_arr_arr)  # 将[[],[],[],...]合并为一个arr
images = images[select_index_arr]
cont_labels = cont_labels[select_index_arr]
class_labels = class_labels[select_index_arr]

# # 保留数据的一个副本（原始数据）
# raw_images = copy.deepcopy(images)
# raw_cont_labels = copy.deepcopy(cont_labels)
# raw_class_labels = copy.deepcopy(class_labels)

# --------------------------- 解决不同标签的样本不平衡问题,删多补少 ---------------------------
print(f"Original set has {len(images)} images \n"
      f"For each label combination, images num should in "
      f"[{MIN_IMG_NUM_PER_LABEL},{MAX_IMG_NUM_PER_LABEL}] \n"
      f"Start solving the problem of sample label imbalance >>>")
# 获取两类标签的唯一有序数组
unique_cont_labels = np.sort(np.array(list(set(cont_labels))))
unique_class_labels = np.sort(np.array(list(set(class_labels))))
assert NUM_CLASSES == len(unique_class_labels)
keep_index_arr_arr = []
replica_index_arr_arr = []
num_log = []
num_log_final = []
for i in range(NUM_CLASSES):
    for j in tqdm(range(len(unique_cont_labels))):
        '''
        在 NumPy 中，如果要在布尔索引表达式里同时满足两个条件（如 class_labels == something 
        并且 cont_labels == something_else），不能直接使用 and，因为它只适用于单个布尔值；
        对布尔数组应当用位运算符 & (注意: &优先级较高, 此处,两侧表达式必须用括号括起来)
        '''
        index_arr = np.where((class_labels == unique_class_labels[i])
                             & (cont_labels == unique_cont_labels[j]))[0]
        num_log.append(len(index_arr))
        # todo: 有可能某些标签组合是缺数据的,待处理
        # assert len(index_arr) != 0, ""
        if len(index_arr) == 0:
            # warnings.warn(f"Label combination [{unique_class_labels[i]},{unique_cont_labels[j]}] "
            #               f"has no data!")  # tqdm 依赖行内刷新,在tqdm内输出内容会显示异常
            tqdm.write(f"UserWarning:Label combination "
                       f"[{unique_class_labels[i]},{unique_cont_labels[j]}] has no data!")
            num_log_final.append(0)
        elif len(index_arr) > MAX_IMG_NUM_PER_LABEL:
            # 如果当前标签样本数量过多，则随机保留指定数量,去除多余的
            np.random.shuffle(index_arr)
            index_arr = index_arr[0:MAX_IMG_NUM_PER_LABEL]
            keep_index_arr_arr.append(index_arr)
            num_log_final.append(MAX_IMG_NUM_PER_LABEL)
        elif len(index_arr) < MIN_IMG_NUM_PER_LABEL:
            # 如果当前标签样本数量过少，则随机复制
            # 已有的先直接保留一份
            keep_index_arr_arr.append(index_arr)
            # 然后随机从当前样本中复制缺少的数量（允许重复）
            num_less = MIN_IMG_NUM_PER_LABEL - len(index_arr)
            index_replica_arr = np.random.choice(index_arr, size=num_less, replace=True)
            replica_index_arr_arr.append(index_replica_arr)
            num_log_final.append(MIN_IMG_NUM_PER_LABEL)
        else:
            num_log_final.append(len(index_arr))

print("View the distribution(img nums) of origin data in each label\n"
      + get_distribution_table(num_log, unique_class_labels, unique_cont_labels))

# 最终的数据
keep_index_arr = np.concatenate(keep_index_arr_arr, axis=0)
replica_index_arr = np.concatenate(replica_index_arr_arr, axis=0)
images = np.concatenate((images[keep_index_arr], images[replica_index_arr]), axis=0)
cont_labels = np.concatenate(
        (cont_labels[keep_index_arr], cont_labels[replica_index_arr]), axis=0)
class_labels = np.concatenate(
        (class_labels[keep_index_arr], class_labels[replica_index_arr]), axis=0)
table_data = []
for row, cls_label in enumerate(unique_class_labels):
    # 当前行对应的 num_log 切片
    row_vals = num_log[row * j: row * j + j]
    # 组合：第一列是 class_label，后面是每个值
    # 注意: if row_vals 是一维 numpy array/list, 需先.tolist()转成 list
    row_data = [cls_label] + row_vals
    table_data.append(row_data)
print("View the distribution(img nums) of final data in each label\n"
      + get_distribution_table(num_log_final, unique_class_labels, unique_cont_labels))
print(f"Finish replication and deletion, final number of pictures: {len(images)} \n")

# --------------------------- 连续标签归一化 ---------------------------
print(f"Range of unNormalized continuous labels: ({np.min(cont_labels)},{np.max(cont_labels)})")
# 使用辅助函数对连续标签归一化到 [0,1]（需要传入最大标签值 max_label）
cont_labels = fn_norm_labels(cont_labels, MAX_LABEL)
print(f"Range of normalized continuous labels: ({np.min(cont_labels)},{np.max(cont_labels)})")
# 获取归一化后唯一的连续标签（用于后续分析或训练数据准备）
unique_cont_labels_norm = np.sort(np.array(list(set(cont_labels))))
print(f"Unique class labels before adjustment:{np.unique(class_labels)}\n")

# --------------------------- 根据数据统计自动计算 kernel_sigma 与 kappa ---------------------------
if KERNEL_SIGMA < 0:
    std_label = np.std(cont_labels)
    KERNEL_SIGMA = 1.06 * std_label * (len(cont_labels)) ** (-1 / 5)
    print("Use rule-of-thumb formula to compute kernel_sigma >>>")
    print(f"The std of {len(cont_labels)} age labels is {std_label} "
          f"so the kernel sigma is {KERNEL_SIGMA}\n")

if KAPPA < 0:
    n_unique = len(unique_cont_labels_norm)
    diff_list = []
    for i in range(1, n_unique):
        diff_list.append(unique_cont_labels_norm[i] - unique_cont_labels_norm[i - 1])
    kappa_base = np.abs(KAPPA) * np.max(np.array(diff_list))
    if THRESHOLD_TYPE == "hard":
        KAPPA = kappa_base
    else:
        KAPPA = 1 / kappa_base ** 2

# --------------------------- 创建输出文件夹 ---------------------------
f_time_str = datetime.now().strftime("%m%d%H")
path_to_output = os.path.join(ROOT_PATH,
                              'output/CcGAN_{}_{}_si{:.3f}_ka{:.3f}_{}_nDs{}_nDa{}_nGa{}_Dbs{}_Gbs{}_v2_{}'
                              .format(GAN_ARCH, THRESHOLD_TYPE, KERNEL_SIGMA, KAPPA, LOSS_TYPE,
                                      NUM_D_STEPS, NUM_GRAD_ACC_D, NUM_GRAD_ACC_G,
                                      BATCH_SIZE_D, BATCH_SIZE_G, f_time_str))
os.makedirs(path_to_output, exist_ok=True)
save_models_folder = os.path.join(path_to_output, 'saved_models')
os.makedirs(save_models_folder, exist_ok=True)
save_images_folder = os.path.join(path_to_output, 'saved_images')
os.makedirs(save_images_folder, exist_ok=True)
# path_to_embed_models = os.path.join(ROOT_PATH, f'output/embed_models_v2_{f_time_str}')
path_to_embed_models = os.path.join(ROOT_PATH, f'output/embed_models_v2_021821')  # 固定该版
os.makedirs(path_to_embed_models, exist_ok=True)

#######################################################################################
'''               Pre-trained CNN and GAN for label embedding                       '''
#######################################################################################
# -------------------- 定义预训练模型的 checkpoint 文件名 --------------------
net_embed_filename_ckpt = os.path.join(path_to_embed_models, 'ckpt_{}_epoch_{}_seed_{}.pth'
                                       .format(NET_EMBED_TYPE, EPOCH_CNN_EMBED, SEED))
net_y2h_filename_ckpt = os.path.join(path_to_embed_models, 'ckpt_net_y2h_epoch_{}_seed_{}.pth'
                                     .format(EPOCH_NET_Y2H, SEED))

# -------------------- 构建训练集和 DataLoader --------------------
# 这里假设 ImgsDataset 类已经支持同时接受三个参数：图像、连续标签、离散标签，并自动归一化图像
train_set = ImgsDataset_v2(images, cont_labels, class_labels, normalize=True)
train_loader_embed_net = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_EMBED,
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
    print("Start training CNN for label embedding >>>")
    net_embed = train_net_embed(net=net_embed, net_name=net_embed,
                                trainloader=train_loader_embed_net,
                                testloader=None, epochs=EPOCH_CNN_EMBED,
                                resume_epoch=RESUME_EPOCH_CNN_EMBED,
                                lr_base=BASE_LR_X2Y, lr_decay_factor=0.1, lr_decay_epochs=[80, 140],
                                weight_decay=1e-4, path_to_ckpt=path_to_embed_models)
    # 保存训练好的 net_embed 模型
    torch.save({
        'net_state_dict': net_embed.state_dict(),
    }, net_embed_filename_ckpt)
else:
    print("net_embed ckpt already exists")
    print("Loading...")
    checkpoint = torch.load(net_embed_filename_ckpt, weights_only=True, map_location=device)
    net_embed.load_state_dict(checkpoint['net_state_dict'])
    print("Loaded successfully.\n")

## (2). 训练 net_y2h：将标签映射到与图像嵌入相同的空间
if not os.path.isfile(net_y2h_filename_ckpt):
    print("Start training net_y2h >>>")
    net_y2h = train_net_y2h(cont_labels, class_labels, net_y2h, net_embed, epochs=EPOCH_NET_Y2H,
                            lr_base=BASE_LR_Y2H, lr_decay_factor=0.1,
                            lr_decay_epochs=[150, 250, 350],
                            weight_decay=1e-4, batch_size=128)
    # 保存训练好的 net_y2h 模型
    torch.save({
        'net_state_dict': net_y2h.state_dict(),
    }, net_y2h_filename_ckpt)
else:
    print("net_y2h ckpt already exists")
    print("Loading...")
    checkpoint = torch.load(net_y2h_filename_ckpt, weights_only=True, map_location=device)
    net_y2h.load_state_dict(checkpoint['net_state_dict'])
    print("Loaded successfully.\n")

## -------------------- 简单测试：检查连续标签映射是否正确 --------------------
net_embed.eval()
net_y2h.eval()
# 从连续标签中随机选择 10 个用于测试映射效果
index_tmp = np.arange(len(unique_cont_labels_norm))
np.random.shuffle(index_tmp)
index_tmp = index_tmp[:10]  # 60个唯一标签索引打乱后,取前10个
# labels_tmp: 形状 (10, 1)，注意这里连续标签已经归一化到 [0,1]
labels_tmp = unique_cont_labels_norm[index_tmp].reshape(-1, 1)
labels_tmp = torch.from_numpy(labels_tmp).type(torch.float).to(device)
# 添加噪声，模拟连续标签的不确定性
epsilons_tmp = np.random.normal(0, 0.2, len(labels_tmp))
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
print("Begin Training >>>")
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
    netG, netD = train_ccgan(KERNEL_SIGMA, KAPPA, images, cont_labels, class_labels, netG, netD,
                             net_y2h,
                             save_images_folder=save_images_in_train_folder,
                             save_models_folder=save_models_folder)
    # 保存训练好的生成器模型
    torch.save({'netG_state_dict': netG.state_dict()}, ckpt_gan_path)
    stop = timeit.default_timer()
    print("GAN training finished; Time elapses: {}s".format(stop - start))
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
    print("Loaded successfully.\n")
