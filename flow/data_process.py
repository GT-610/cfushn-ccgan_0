import copy

import h5py

from config import cfg
from utils.data_util import show_class_labels_map
from utils.utils import *

device = cfg.device
np_rng = np.random.default_rng(cfg.seed)


def data_process():
    """数据预处理"""
    # --------------------------- 加载数据 ---------------------------
    # 数据文件名：根据图像尺寸构造 h5 文件名（例如 UTKFace_64x64.h5）
    data_filename = f"{cfg.data_path}/{cfg.dataset_name}_{cfg.img_size}x{cfg.img_size}.h5"
    hf = h5py.File(data_filename, 'r')

    # 加载图像数据
    images = hf[cfg.image_set_h5_key][:]
    # 加载连续标签（例如年龄），并转为 float 类型
    cont_labels = (hf[cfg.cont_label_h5_key][:]).astype(float)  # numpy.ndarray
    # 加载离散标签（注意:非下标!
    class_labels = (hf[cfg.class_label_h5_key][:]).astype(str)  # numpy.ndarray
    hf.close()

    # # --------------------------- 数据子集选择 scoping ---------------------------
    # # 根据连续标签的范围 [min_label, max_label] 筛选数据
    # selected_cont_labels = np.arange(cfg.min_label, cfg.max_label + 1)
    # select_index_arr_arr = []  # [[],[],[],...]
    # for i in range(len(selected_cont_labels)):
    #     curr_cont = selected_cont_labels[i]
    #     # 找出连续值等于当前值的所有样本索引(只取整数标签的,否则没有研究意义)
    #     index_curr = np.where(cont_labels == curr_cont)[0]  # (用到广播机制)
    #     select_index_arr_arr.append(index_curr)
    # # 更新数据集：只保留所选子集
    # select_index_arr = np.concatenate(select_index_arr_arr)  # 将[[],[],[],...]合并为一个arr
    # images = images[select_index_arr]
    # cont_labels = cont_labels[select_index_arr]
    # class_labels = class_labels[select_index_arr]

    # 确保cont_labels始终为(N,cont_dim)格式,因为numpy会自动把长度=1的维度省略掉
    # 例如(N,1)会变成(N,),所以要用reshape强制保持二维
    cont_labels = cont_labels.reshape(len(cont_labels), -1)
    assert cfg.cont_dim == cont_labels.shape[1], \
        f"cont_dim{cfg.cont_dim} must be corresponding to the cont_labels{cont_labels.shape[1]}"
    unique_class_labels = np.sort(np.unique(class_labels))
    assert cfg.num_classes == len(unique_class_labels), \
        f"{cfg.num_classes}!={len(unique_class_labels)}!!"
    # 数据集中存储的是类别标签, 而非类别标签索引! 计算交叉熵时候必须用索引!
    # 为了简化实验,仅在数据处理时打印一下对应关系,后续的class_labels存储的都是索引!
    # reverse_label_mapping为{"class_name":class_index}
    reverse_label_mapping = show_class_labels_map(unique_class_labels)
    # class_labels转换为类索引数组
    class_labels = [reverse_label_mapping[class_name] for class_name in class_labels]
    class_labels = np.array(class_labels)

    # 保留数据的一个副本（原始数据）
    raw_images = copy.deepcopy(images)
    raw_cont_labels = copy.deepcopy(cont_labels)
    raw_class_labels = copy.deepcopy(class_labels)

    # # --------------------------- 解决不同标签的样本不平衡问题,删多补少 ---------------------------
    # print(f"Original set has {len(images)} images \n"
    #       f"For each label combination, images num should in "
    #       f"[{cfg.min_img_num_per_label},{cfg.max_img_num_per_label}] \n"
    #       f"Start solving the problem of sample label imbalance >>>")
    # print("num of classes: ", cfg.num_classes)
    # keep_index_arr_arr = []
    # replica_index_arr_arr = []
    # num_log = []
    # num_log_final = []
    # for i in range(cfg.num_classes):
    #     for j in tqdm(range(len(unique_cont_labels))):
    #         index_arr = np.where((class_labels == unique_class_labels[i])
    #                              * (cont_labels == unique_cont_labels[j]))[0]
    #         num_log.append(len(index_arr))
    #         # todo: 有可能某些标签组合是缺数据的,待处理
    #         # assert len(index_arr) != 0, ""
    #         if len(index_arr) == 0:
    #             # warnings.warn(f"Label combination [{unique_class_labels[i]},{unique_cont_labels[j]}] "
    #             #               f"has no data!")  # tqdm 依赖行内刷新,在tqdm内输出内容会显示异常
    #             tqdm.write(f"UserWarning:Label combination "
    #                        f"[{unique_class_labels[i]},{unique_cont_labels[j]}] has no data!")
    #             num_log_final.append(0)
    #         elif len(index_arr) > cfg.max_img_num_per_label:
    #             # 如果当前标签样本数量过多，则随机保留指定数量,去除多余的
    #             np_rng.shuffle(index_arr)
    #             index_arr = index_arr[0:cfg.max_img_num_per_label]
    #             keep_index_arr_arr.append(index_arr)
    #             num_log_final.append(cfg.max_img_num_per_label)
    #         elif len(index_arr) < cfg.min_img_num_per_label:
    #             # 如果当前标签样本数量过少，则随机复制
    #             # 已有的先直接保留一份
    #             keep_index_arr_arr.append(index_arr)
    #             # 然后随机从当前样本中复制缺少的数量（允许重复）
    #             num_less = cfg.min_img_num_per_label - len(index_arr)
    #             index_replica_arr = np_rng.choice(index_arr, size=num_less, replace=True)
    #             replica_index_arr_arr.append(index_replica_arr)
    #             num_log_final.append(cfg.min_img_num_per_label)
    #         else:
    #             keep_index_arr_arr.append(index_arr)
    #             num_log_final.append(len(index_arr))
    # # 最终的数据
    # keep_index_arr = np.concatenate(keep_index_arr_arr, axis=0)
    # if replica_index_arr_arr:
    #     replica_index_arr = np.concatenate(replica_index_arr_arr, axis=0)
    #     images = np.concatenate((images[keep_index_arr], images[replica_index_arr]), axis=0)
    #     cont_labels = np.concatenate(
    #             (cont_labels[keep_index_arr], cont_labels[replica_index_arr]), axis=0)
    #     class_labels = np.concatenate(
    #             (class_labels[keep_index_arr], class_labels[replica_index_arr]), axis=0)
    # else:
    #     images = images[keep_index_arr]
    #     cont_labels = cont_labels[keep_index_arr]
    #     class_labels = class_labels[keep_index_arr]

    # # 打印不平衡处理前后的分布对比
    # print("View the distribution(img nums) of origin data in each label\n"
    #       + get_distribution_table(num_log, unique_class_labels, unique_cont_labels))
    # print("View the distribution(img nums) of final data in each label\n"
    #       + get_distribution_table(num_log_final, unique_class_labels, unique_cont_labels))
    print(f"Finish replication and deletion, final number of pictures: {len(images)}"
          f"(origin:{len(raw_images)}) \n")

    # --------------------------- 对连续标签归一化 ---------------------------
    print(f"Range of unNormalized continuous labels: "
          f"({np.min(cont_labels, axis=0)},{np.max(cont_labels, axis=0)})")
    # 使用辅助函数对连续标签归一化到 [0,1]（需要传入最大标签值 max_label）
    cont_labels = fn_norm_labels(cont_labels, cfg.max_label)
    print(f"Range of normalized continuous labels: "
          f"({np.min(cont_labels, axis=0)},{np.max(cont_labels, axis=0)})")
    print(f"Unique class labels before adjustment:{np.unique(class_labels)}\n")

    # --------------------------- 根据数据统计自动计算 kernel_sigma 与 kappa ---------------------------
    compute_sigma_kappa(cont_labels)

    return [raw_images, raw_cont_labels, raw_class_labels], [images, cont_labels, class_labels]


def compute_sigma_kappa(cont_labels) -> None:
    """
    计算核函数带宽 kernel_sigma 和邻域参数 kappa
    适用于 (N, cont_dim) 形状的高维连续标签
    """

    # --------------------------- 计算 kernel_sigma ---------------------------
    std_label_per_dim = np.std(cont_labels, axis=0)  # shape: (cont_dim,)
    cfg.kernel_sigma = 1.06 * std_label_per_dim * (len(cont_labels)) ** (-1 / 5)
    print("Use rule-of-thumb formula to compute kernel_sigma >>>\n"
          f"The std of {len(cont_labels)} labels is {std_label_per_dim},"
          f"so the kernel sigma is {cfg.kernel_sigma}")

    # --------------------------- 计算 kappa ---------------------------
    """
    对于计算kappa, 应对多维标签, 有两种方案.
    方案1: kappa也改成相应的多维的, 对每个维度分别计算kappa[d]=max(list_d[diff_dist]),kappa为(d,)
    方案2: 计算所有标签组成的K近邻树,取每个样本的第二近邻(即除自身外最近者)的距离,然后再取max,这是一个标量
    这里选择方案1
    """
    # 对每个维度分别计算kappa
    # 直接将cont_labels转换为(N, cont_dim)形状
    cont_labels_dims = cont_labels.shape[1]  # 获取标签的维度数
    kappa_values = np.zeros(cont_labels_dims)

    for d in range(cont_labels_dims):
        # 提取当前维度的所有标签值
        dim_labels = cont_labels[:, d]
        # 计算相邻标签之间的差异
        sorted_labels = np.sort(dim_labels)
        diff_dist = np.diff(sorted_labels)
        # 对每个维度计算kappa
        if len(diff_dist) > 0:
            kappa_values[d] = np.max(diff_dist)
        else:
            kappa_values[d] = 1.0  # 默认值，当只有一个样本时

    kappa_base = np.abs(cfg.kappa) * kappa_values

    if cfg.threshold_type == "hard":
        cfg.kappa = kappa_base
    else:
        # 对于软阈值，需要对每个维度分别计算
        cfg.kappa = 1 / kappa_base ** 2
        # 防止除0出现无穷大
        cfg.kappa = np.where(np.isinf(cfg.kappa), 1e6, cfg.kappa)



    print(f"Computed kappa_base: {kappa_base}, Final {cfg.threshold_type} kappa: {cfg.kappa}\n")
    # 此处的cfg.kappa类型为(cont_dim,), 是numpy.ndarray类型
