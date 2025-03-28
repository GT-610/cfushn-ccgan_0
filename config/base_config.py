from datetime import datetime


class BaseConfig:
    # 基本配置信息
    version = "v" + datetime.now().strftime("%m%d%H")  # 版本号
    device = "cuda"
    root_path = "./"

    # 数据集相关
    dataset_name = "UTKFace"  # 数据集名称, eg：UTKFace, RC-49
    image_set_h5_key = "images"  # 图片集在h5中的字典key
    cont_label_h5_key = "labels"  # 连续标签在h5中的字典key eg: label (UTKFace)
    class_label_h5_key = "races"  # 离散标签在h5中的字典key eg: races (UTKFace),races (RC-49)
    num_classes: int = 5  # 类别数量（离散标签）# !类别数是对于一个数据集而言是固定的! 测试和验证集都不能越界!
    img_size: int = 64  # 宽=高=img_size
    cont_dim: int = 1  # 连续标签的维度(离散标签有几种)
    kernel_sigma: [float] = [-1.0]  # 高斯核的标准差
    kappa: [float] = [-1.0]  # 邻域半径

    # 训练环境
    seed: int = 2025  # 随机种子
    gpu_parallel = True  # 是否使用多 GPU 并行训练(如果指定CUDA_VISIBLE_DEVICES为单个,则该选项无效)
    num_workers: int = 0  # 数据加载时的线程数
    n_iters: int = 20000  # 训练的总迭代次数
    resume_n_iters: int = 1  # 训练恢复起始iter (0:从头训练,>0:从指定迭代次数或最新ckpt(若均无,则从头训练))
    save_n_iters_freq: int = 2000  # 模型 checkpoint 的保存频率（迭代步数）
    visualize_freq: int = 500  # 可视化生成图像的频率（迭代步数）

    # 数据处理相关
    min_img_num_per_label: int = 0  # 每个标签最少样本数（不足则复制）解决数据不平衡
    max_img_num_per_label: int = 99999  # 每个标签最多样本数（超出则随机删除）解决数据不平衡
    min_label: [float] = [0.0]  # 连续标签最小值,用于数据筛选与归一化
    max_label: [float] = [90.0]  # 连续标签最大值,用于数据筛选与归一化

    # 生成器 & 嵌入空间
    dim_gan: int = 256  # 生成器输入噪声的维度
    dim_embed: int = 128  # 嵌入空间的维度
    net_embed_type = "ResNet34_embed"  # 生成器使用的嵌入模型
    base_lr_x2y: float = 0.01  # net_x2y（图像到嵌入）的学习率
    base_lr_y2h: float = 0.01  # net_y2h（标签到嵌入）的学习率

    # 训练超参数
    num_d_steps: int = 2  # 每次训练中判别器更新的步数
    lr_g: float = 1e-4  # 生成器学习率
    lr_d: float = 1e-4  # 判别器学习率
    batch_size_d: int = 256  # 判别器的 batch size
    batch_size_g: int = 256  # 生成器的 batch size
    num_grad_acc_d: int = 1  # 判别器梯度累积步数
    num_grad_acc_g: int = 1  # 生成器梯度累积步数

    # 训练目标 & 损失函数
    gan_arch = "SNGAN"  # GAN 结构类型
    loss_type = "vanilla"  # 损失类型 vanilla,hinge
    threshold_type = "soft"  # 邻域阈值类型（'hard' 或 'soft'）
    nonzero_soft_weight_threshold: float = 1e-3  # 软阈值下的非零权重最小值,用于SVDL损失计算

    # CNN 训练参数
    epoch_cnn_embed: int = 200  # net_embed 训练的总 Epochs
    resume_epoch_cnn_embed: int = 0  # 继续训练的起始 Epoch
    epoch_net_y2h: int = 500  # net_y2h 训练的总 Epochs
    batch_size_embed: int = 256  # net_embed 训练的 batch size

    # 数据增强
    use_DiffAugment = True  # 是否使用 DiffAugment 数据增强技术
    policy = "translation,cutout"  # 采用的数据增强策略（可选：'color,translation,cutout'）

    # 采样
    nrow: int = 20  # 采样图像网格每行多少个 (一个类别一行)
    samp_batch_size: int = 200  # 采样时的 batch size

    # 评估
    if_eval = False  # 是否评估
    pretrained_ae_pth = "ckpt_AE_epoch_200_seed_2020_CVMode_False.pth"
    pretrained_cnn4cont_pth = "ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_200_seed_2020_classify_5_races_CVMode_False.pth"
    pretrained_cnn4class_pth = "ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth"
    n_fake_per_label: int = 200  # 每个连续标签(整数)生成多少张图像用于评估
    epoch_ae: int = 200
    epoch_fid_cnn: int = 200  # 计算 FID 时使用的 CNN 训练 Epoch
    fid_radius: int = 0  # FID 计算时的邻域半径
    dump_fake_for_niqe = True  # 是否导出用于 NIQE 计算的图像
    comp_is_and_fid_only = False  # 是否只计算 IS 和 FID（减少计算量）

    # 目录路径
    @property
    def data_path(self) -> str:
        return f"{self.root_path}/data"

    @property
    def torch_model_path(self) -> str:
        return f"{self.root_path}/pretrained"  # PyTorch 预训练模型路径

    @property
    def eval_path(self) -> str:
        return f"{self.torch_model_path}/eval/{self.dataset_name}"

    @property
    def output_path(self) -> str:
        return f'{self.root_path}/output/{self.dataset_name}_{self.img_size}x{self.img_size}'

    @property
    def version_output_path(self) -> str:
        return f'{self.output_path}/{self.version}'

    @property
    def niqe_dump_path(self) -> str:  # NIQE 计算的伪造数据存储路径
        # return f"{self.output_path}/NIQE_{self.img_size}x{self.img_size}"
        return f"{self.version_output_path}/saved_images/fake_images"

    def pretty_str(self) -> str:
        """一行一行打印参数"""
        pretty = ""
        # for key, value in self.__dict__.items():
        #     pretty += (f"{key}: {value}\n" if key != "_instance" else "")
        # 上面的方法无法保证属性是有序输出的
        for key in self.__annotations__.keys():
            pretty += (f"{key}: {getattr(self, key)}\n" if key != "_instance" else "")
        pretty += f"data_path: {self.data_path}\n"
        pretty += f"torch_model_path: {self.torch_model_path}\n"
        pretty += f"eval_path: {self.eval_path}\n"
        pretty += f"output_path: {self.output_path}\n"
        pretty += f"gan_output_path: {self.version_output_path}\n"
        pretty += f"niqe_dump_path: {self.niqe_dump_path}"
        return pretty

    # 配置类,采用单例模式
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(BaseConfig, cls).__new__(cls)
        return cls._instance


# 直接创建默认配置（单例）
cfg = BaseConfig.__new__(BaseConfig)

if __name__ == "__main__":
    print(cfg.pretty_str())
