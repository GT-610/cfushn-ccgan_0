import copy

import h5py  # 读取 HDF5 文件
from tqdm import tqdm  # 显示进度条

from config.config import *
from models.ResNet_embed import *
from utils.data_util import get_distribution_table
from utils.utils import *


def data_process():
    """数据预处理"""
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

    # 保留数据的一个副本（原始数据）
    raw_images = copy.deepcopy(images)
    raw_cont_labels = copy.deepcopy(cont_labels)
    raw_class_labels = copy.deepcopy(class_labels)

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
                keep_index_arr_arr.append(index_arr)
                num_log_final.append(len(index_arr))
    # 最终的数据
    keep_index_arr = np.concatenate(keep_index_arr_arr, axis=0)
    replica_index_arr = np.concatenate(replica_index_arr_arr, axis=0)
    images = np.concatenate((images[keep_index_arr], images[replica_index_arr]), axis=0)
    cont_labels = np.concatenate(
            (cont_labels[keep_index_arr], cont_labels[replica_index_arr]), axis=0)
    class_labels = np.concatenate(
            (class_labels[keep_index_arr], class_labels[replica_index_arr]), axis=0)
    # 打印不平衡处理前后的分布对比
    print("View the distribution(img nums) of origin data in each label\n"
          + get_distribution_table(num_log, unique_class_labels, unique_cont_labels))
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
    std_label = np.std(cont_labels)
    kernel_sigma = 1.06 * std_label * (len(cont_labels)) ** (-1 / 5)
    print("Use rule-of-thumb formula to compute kernel_sigma >>>")
    print(f"The std of {len(cont_labels)} age labels is {std_label} "
          f"so the kernel sigma is {kernel_sigma}")

    kappa = -1.0
    n_unique = len(unique_cont_labels_norm)
    diff_list = []
    for i in range(1, n_unique):
        diff_list.append(unique_cont_labels_norm[i] - unique_cont_labels_norm[i - 1])
    kappa_base = np.abs(kappa) * np.max(np.array(diff_list))
    if THRESHOLD_TYPE == "hard":
        kappa = kappa_base
    else:
        kappa = 1 / kappa_base ** 2
    print(f"The kappa is {kappa}\n")

    return ([raw_images, raw_cont_labels, raw_class_labels],
            [images, cont_labels, class_labels],
            kernel_sigma, kappa)
