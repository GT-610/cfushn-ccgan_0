from utils.constants import device

print("\n========================================================================================")

# -------------------- 导入第三方包 --------------------
import copy  # 深拷贝对象
import gc  # 垃圾回收工具

import matplotlib.pyplot as plt  # 绘图

plt.switch_backend('agg')  # 使用非交互式后端
import h5py  # 读取 HDF5 文件
import os  # 文件与目录操作
import random  # 随机数生成
from tqdm import tqdm  # 显示进度条
import torch.backends.cudnn as cudnn  # cuDNN 设置
import timeit  # 计时工具

# -------------------- 导入项目内部模块 --------------------
from opts import parse_opts  # 解析命令行参数的工具函数

args = parse_opts()  # 解析所有命令行参数
wd = args.root_path  # 获取项目的根路径
os.chdir(wd)  # 切换到根路径
from utils.utils import *  # 导入项目常用工具函数
from utils.log_util import cy_log
from utils.ipc_util import register_signal_handler, get_s1, get_s2
from models import *  # 导入项目中定义的各种模型
from train_ccgan import train_ccgan  # 导入 GAN 训练及采样函数
from train_net_for_label_embed import train_net_embed, train_net_y2h

#######################################################################################
'''                                   Settings                                      '''
#######################################################################################

# -------------------- 注册信号事件 --------------------
register_signal_handler()
# -------------------- 定义trap,根据信号执行操作 --------------------
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

# -------------------- 设置随机种子，确保结果可复现 --------------------
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)

# -------------------- Embedding 部分超参数 --------------------
base_lr_x2y = 0.01  # 用于训练 net_embed（图像到嵌入）的基础学习率
base_lr_y2h = 0.01  # 用于训练 net_y2h（标签到嵌入）的基础学习率

NGPU = torch.cuda.device_count()  # 当前可用的 GPU 数量

# 如果指定了 torch 模型保存路径，则设置环境变量 TORCH_HOME
if args.torch_model_path != "None":
    os.environ['TORCH_HOME'] = args.torch_model_path


# -------------------- 工具函数：标签归一化与反归一化 --------------------
def fn_norm_labels(labels):
    """
    将未归一化的标签转换到 [0,1] 区间

    参数:
        labels (np.ndarray): 原始标签数组

    返回:
        np.ndarray: 归一化后的标签数组（除以 args.max_label）
    """
    return labels / args.max_label


def fn_denorm_labels(labels):
    """
    将归一化的标签还原为原始尺度

    参数:
        labels (np.ndarray 或 torch.Tensor 或数字): 归一化后的标签

    返回:
        与输入类型对应的标签，数值范围恢复到 [0, args.max_label]
    """
    if isinstance(labels, np.ndarray):
        return (labels * args.max_label).astype(int)
    elif torch.is_tensor(labels):
        return (labels * args.max_label).type(torch.int)
    else:
        return int(labels * args.max_label)


#######################################################################################
'''                                    Data loader                                 '''
#######################################################################################
# -------------------- 加载数据 --------------------
# 数据文件名：根据图像尺寸构造 h5 文件名（例如 UTKFace_64x64.h5）
data_filename = args.data_path + '/UTKFace_{}x{}.h5'.format(args.img_size, args.img_size)
hf = h5py.File(data_filename, 'r')
labels = hf['labels'][:]  # 加载标签数据
labels = labels.astype(float)  # 转为浮点型
images = hf['images'][:]  # 加载图像数据
hf.close()

# -------------------- 数据子集选择 --------------------
# 选取指定标签范围 [min_label, max_label]
selected_labels = np.arange(args.min_label, args.max_label + 1)
images_subset = None
labels_subset = None
for i in range(len(selected_labels)):
    curr_label = selected_labels[i]
    index_curr_label = np.where(labels == curr_label)[0]
    if i == 0:
        images_subset = images[index_curr_label]
        labels_subset = labels[index_curr_label]
    else:
        images_subset = np.concatenate((images_subset, images[index_curr_label]), axis=0)
        labels_subset = np.concatenate((labels_subset, labels[index_curr_label]))
images = images_subset
labels = labels_subset
del images_subset, labels_subset
gc.collect()

# 保留数据的一个副本（原始数据）
raw_images = copy.deepcopy(images)
raw_labels = copy.deepcopy(labels)

# -------------------- 每个标签最多保留指定数量的图像 --------------------
image_num_threshold = args.max_num_img_per_label
print("\n Original set has {} images; For each label, take no more than {} images>>>".format(
        len(images), image_num_threshold))
unique_labels_tmp = np.sort(np.array(list(set(labels))))
sel_indx = None
for i in tqdm(range(len(unique_labels_tmp))):
    indx_i = np.where(labels == unique_labels_tmp[i])[0]
    if len(indx_i) > image_num_threshold:
        np.random.shuffle(indx_i)
        indx_i = indx_i[0:image_num_threshold]
    if i == 0:
        sel_indx = indx_i
    else:
        sel_indx = np.concatenate((sel_indx, indx_i))
images = images[sel_indx]
labels = labels[sel_indx]
print("{} images left.".format(len(images)))

# -------------------- 复制少数样本以缓解类别不平衡 --------------------
max_num_img_per_label_after_replica = np.min(
        [args.max_num_img_per_label_after_replica, args.max_num_img_per_label])
if max_num_img_per_label_after_replica > 1:
    unique_labels_replica = np.sort(np.array(list(set(labels))))
    num_labels_replicated = 0
    print("Start replicating minority samples >>>")
    images_replica = None
    labels_replica = None
    for i in tqdm(range(len(unique_labels_replica))):
        curr_label = unique_labels_replica[i]
        indx_i = np.where(labels == curr_label)[0]
        if len(indx_i) < max_num_img_per_label_after_replica:
            num_img_less = max_num_img_per_label_after_replica - len(indx_i)
            indx_replica = np.random.choice(indx_i, size=num_img_less, replace=True)
            if num_labels_replicated == 0:
                images_replica = images[indx_replica]
                labels_replica = labels[indx_replica]
            else:
                images_replica = np.concatenate((images_replica, images[indx_replica]), axis=0)
                labels_replica = np.concatenate((labels_replica, labels[indx_replica]))
            num_labels_replicated += 1
    images = np.concatenate((images, images_replica), axis=0)
    labels = np.concatenate((labels, labels_replica))
    print("We replicate {} images and labels \n".format(len(images_replica)))
    del images_replica, labels_replica
    gc.collect()

# -------------------- 标签归一化 --------------------
print("\n Range of unnormalized labels: ({},{})".format(np.min(labels), np.max(labels)))
labels = fn_norm_labels(labels)
print("\n Range of normalized labels: ({},{})".format(np.min(labels), np.max(labels)))
unique_labels_norm = np.sort(np.array(list(set(labels))))

# -------------------- 根据数据统计自动计算 kernel_sigma 与 kappa --------------------
if args.kernel_sigma < 0:
    std_label = np.std(labels)
    args.kernel_sigma = 1.06 * std_label * (len(labels)) ** (-1 / 5)
    print("\n Use rule-of-thumb formula to compute kernel_sigma >>>")
    print("\n The std of {} labels is {} so the kernel sigma is {}".format(len(labels), std_label,
                                                                           args.kernel_sigma))

if args.kappa < 0:
    n_unique = len(unique_labels_norm)
    diff_list = []
    for i in range(1, n_unique):
        diff_list.append(unique_labels_norm[i] - unique_labels_norm[i - 1])
    kappa_base = np.abs(args.kappa) * np.max(np.array(diff_list))
    if args.threshold_type == "hard":
        args.kappa = kappa_base
    else:
        args.kappa = 1 / kappa_base ** 2

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
'''               Pre-trained CNN and GAN for label embedding                       '''
#######################################################################################
# -------------------- 定义预训练模型的 checkpoint 文件名 --------------------
net_embed_filename_ckpt = os.path.join(path_to_embed_models, 'ckpt_{}_epoch_{}_seed_{}.pth'.format(
        args.net_embed, args.epoch_cnn_embed, args.seed))
net_y2h_filename_ckpt = os.path.join(path_to_embed_models,
                                     'ckpt_net_y2h_epoch_{}_seed_{}.pth'.format(
                                             args.epoch_net_y2h, args.seed))

print("\n " + net_embed_filename_ckpt)
print("\n " + net_y2h_filename_ckpt)

# -------------------- 构建训练集和 DataLoader --------------------
trainset = ImgsDataset(images, labels, normalize=True)
trainloader_embed_net = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_embed,
                                                    shuffle=True, num_workers=args.num_workers)

# -------------------- 构建图像嵌入模型 net_embed --------------------
net_embed = None
if args.net_embed == "ResNet18_embed":
    net_embed = ResNet18_embed(dim_embed=args.dim_embed)
elif args.net_embed == "ResNet34_embed":
    net_embed = ResNet34_embed(dim_embed=args.dim_embed)
elif args.net_embed == "ResNet50_embed":
    net_embed = ResNet50_embed(dim_embed=args.dim_embed)
net_embed = net_embed.to(device)

# -------------------- 构建标签映射模型 net_y2h --------------------
net_y2h = model_y2h(dim_embed=args.dim_embed)
net_y2h = net_y2h.to(device)

## (1). 训练 net_embed：将图像映射到嵌入空间，然后通过 h2y 映射回标签（x2h+h2y）
if not os.path.isfile(net_embed_filename_ckpt):
    print("\n Start training CNN for label embedding >>>")
    net_embed = train_net_embed(net=net_embed, net_name=args.net_embed,
                                trainloader=trainloader_embed_net,
                                testloader=None, epochs=args.epoch_cnn_embed,
                                resume_epoch=args.resumeepoch_cnn_embed,
                                lr_base=base_lr_x2y, lr_decay_factor=0.1, lr_decay_epochs=[80, 140],
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
    net_y2h = train_net_y2h(unique_labels_norm, net_y2h, net_embed, epochs=args.epoch_net_y2h,
                            lr_base=base_lr_y2h, lr_decay_factor=0.1,
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

## -------------------- 简单测试，检查标签映射是否正确 --------------------
unique_labels_norm_embed = np.sort(np.array(list(set(labels))))
indx_tmp = np.arange(len(unique_labels_norm_embed))
np.random.shuffle(indx_tmp)
indx_tmp = indx_tmp[:10]
labels_tmp = unique_labels_norm_embed[indx_tmp].reshape(-1, 1)
labels_tmp = torch.from_numpy(labels_tmp).type(torch.float).to(device)
epsilons_tmp = np.random.normal(0, 0.2, len(labels_tmp))
epsilons_tmp = torch.from_numpy(epsilons_tmp).view(-1, 1).type(torch.float).to(device)
labels_noise_tmp = torch.clamp(labels_tmp + epsilons_tmp, 0.0, 1.0)
net_embed.eval()
net_h2y = net_embed.h2y
net_y2h.eval()
with torch.no_grad():
    labels_hidden_tmp = net_y2h(labels_tmp)
    labels_noise_hidden_tmp = net_y2h(labels_noise_tmp)
    labels_rec_tmp = net_h2y(labels_hidden_tmp).cpu().numpy().reshape(-1, 1)
    labels_noise_rec_tmp = net_h2y(labels_noise_hidden_tmp).cpu().numpy().reshape(-1, 1)
    labels_hidden_tmp = labels_hidden_tmp.cpu().numpy()
    labels_noise_hidden_tmp = labels_noise_hidden_tmp.cpu().numpy()
labels_tmp = labels_tmp.cpu().numpy()
labels_noise_tmp = labels_noise_tmp.cpu().numpy()
results1 = np.concatenate((labels_tmp, labels_rec_tmp), axis=1)
print("\n labels vs reconstructed labels")
print(results1)
labels_diff = (labels_tmp - labels_noise_tmp) ** 2
hidden_diff = np.mean((labels_hidden_tmp - labels_noise_hidden_tmp) ** 2, axis=1, keepdims=True)
results2 = np.concatenate((labels_diff, hidden_diff), axis=1)
print("\n labels diff vs hidden diff")
print(results2)

# 将嵌入模型放回 CPU 并释放内存
net_embed = net_embed.cpu()
net_h2y = net_h2y.cpu()
del net_embed, net_h2y
gc.collect()
net_y2h = net_y2h.cpu()

#######################################################################################
'''                                    GAN training                                 '''
#######################################################################################
print("CcGAN: {}, {}, Sigma is {:.4f}, Kappa is {:.4f}.".format(args.GAN_arch, args.threshold_type,
                                                                args.kernel_sigma, args.kappa))
save_images_in_train_folder = os.path.join(save_images_folder, 'images_in_train')
os.makedirs(save_images_in_train_folder, exist_ok=True)

start = timeit.default_timer()
print("\n Begin Training >>>")
ckpt_gan_path = os.path.join(save_models_folder, 'ckpt_niter_{}.pth'.format(args.niters_gan))
print(ckpt_gan_path)
netG = None
netD = None
if not os.path.isfile(ckpt_gan_path):
    # 根据 GAN 架构选择生成器与判别器
    if args.GAN_arch == "SAGAN":
        # netG = sagan_generator(nz=args.dim_gan, dim_embed=args.dim_embed).to(device)
        # netD = sagan_discriminator(dim_embed=args.dim_embed).to(device)
        pass
    else:
        netG = sngan_generator(nz=args.dim_gan, dim_embed=args.dim_embed).to(device)
        netD = sngan_discriminator(dim_embed=args.dim_embed).to(device)
    netG = nn.DataParallel(netG)  # 使用多GPU并行训练
    netD = nn.DataParallel(netD)

    # 调用 train_ccgan 函数进行 GAN 训练
    netG, netD = train_ccgan(args.kernel_sigma, args.kappa, images, labels, netG, netD, net_y2h,
                             save_images_folder=save_images_in_train_folder,
                             save_models_folder=save_models_folder)
    # 保存训练好的生成器模型
    torch.save({'netG_state_dict': netG.state_dict()}, ckpt_gan_path)
else:
    print("Loading pre-trained generator >>>")
    checkpoint = torch.load(ckpt_gan_path, weights_only=True, map_location=device)
    # 根据 GAN 架构选择生成器
    if args.GAN_arch == "SAGAN":
        # netG = sagan_generator(nz=args.dim_gan, dim_embed=args.dim_embed).to(device)
        pass
    else:
        netG = sngan_generator(nz=args.dim_gan, dim_embed=args.dim_embed).to(device)
    netG = nn.DataParallel(netG)
    netG.load_state_dict(checkpoint['netG_state_dict'])

stop = timeit.default_timer()
print("GAN training finished; Time elapses: {}s".format(stop - start))
