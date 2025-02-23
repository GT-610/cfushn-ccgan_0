from datetime import datetime
from typing import ClassVar

from pydantic import BaseModel


class BaseConfig(BaseModel):
    # 基本配置信息
    version: str = "v" + datetime.now().strftime("%m%d%H")  # 版本号
    device: str = "cuda"
    root_path: str = "/home/cy/workdir/cfushn-ccgan_0"

    # 数据集相关
    dataset_name: str = "UTKFace"  # 数据集名称, eg：UTKFace, RC-49
    image_set_h5_key: str = "images"  # 图片集在h5中的字典key
    cont_label_h5_key: str = "label"  # 连续标签在h5中的字典key eg: label (UTKFace)
    class_label_h5_key: str = "races"  # 离散标签在h5中的字典key eg: races (UTKFace),races (RC-49)
    num_classes: int = 5  # 类别数量（离散标签）# !类别数是对于一个数据集而言是固定的! 测试和验证集都不能越界!
    img_size: int = 64  # 宽=高=img_size
    kernel_sigma: float = -1.0
    kappa: float = -1.0

    # 训练环境
    seed: int = 2025  # 随机种子
    gpu_parallel: bool = True  # 是否使用多 GPU 并行训练(如果指定CUDA_VISIBLE_DEVICES为单个,则该选项无效)
    num_workers: int = 0  # 数据加载时的线程数
    n_iters: int = 40000  # 训练的总迭代次数
    resume_n_iters: int = 1  # 训练恢复起始iter (0:从头训练,>0:从指定迭代次数或最新ckpt(若均无,则从头训练))
    save_n_iters_freq: int = 2000  # 模型 checkpoint 的保存频率（迭代步数）
    visualize_freq: int = 500  # 可视化生成图像的频率（迭代步数）

    # 数据处理相关
    min_img_num_per_label: int = 0  # 每个标签最少样本数（不足则复制）解决数据不平衡
    max_img_num_per_label: int = 99999  # 每个标签最多样本数（超出则随机删除）解决数据不平衡
    min_label: float = 1.0  # 连续标签最小值,用于数据筛选与归一化
    max_label: float = 60.0  # 连续标签最大值,用于数据筛选与归一化

    # 生成器 & 嵌入空间
    dim_gan: int = 256  # 生成器输入噪声的维度
    dim_embed: int = 128  # 嵌入空间的维度
    net_embed_type: str = "ResNet34_embed"  # 生成器使用的嵌入模型
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
    gan_arch: str = "SNGAN"  # GAN 结构类型
    loss_type: str = "vanilla"  # 损失类型 vanilla,hinge
    threshold_type: str = "soft"  # 邻域阈值类型（'hard' 或 'soft'）
    nonzero_soft_weight_threshold: float = 1e-3  # 软阈值下的非零权重最小值,用于SVDL损失计算

    # CNN 训练参数
    epoch_cnn_embed: int = 200  # net_embed 训练的总 Epochs
    resume_epoch_cnn_embed: int = 0  # 继续训练的起始 Epoch
    epoch_net_y2h: int = 500  # net_y2h 训练的总 Epochs
    batch_size_embed: int = 256  # net_embed 训练的 batch size

    # 数据增强
    use_DiffAugment: bool = True  # 是否使用 DiffAugment 数据增强技术
    policy: str = "translation,cutout"  # 采用的数据增强策略（可选：'color,translation,cutout'）

    # 采样
    nrow: int = 20  # 采样图像网格的行数
    samp_batch_size: int = 200  # 采样时的 batch size

    # 评估
    pretrained_ae_pth: str = "ckpt_AE_epoch_200_seed_2020_CVMode_False.pth"
    pretrained_cnn4cont_pth: str = "ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_200_seed_2020_classify_5_races_CVMode_False.pth"
    pretrained_cnn4class_pth: str = "ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth"
    n_fake_per_label: int = 200  # 每个连续标签(整数)生成多少张图像用于评估
    epoch_ae: int = 200
    comp_fid: bool = False  # 是否计算 FID 分数
    epoch_fid_cnn: int = 200  # 计算 FID 时使用的 CNN 训练 Epoch
    fid_radius: int = 0  # FID 计算时的邻域半径
    dump_fake_for_niqe: bool = True  # 是否导出用于 NIQE 计算的图像
    comp_is_and_fid_only: bool = False  # 是否只计算 IS 和 FID（减少计算量）

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
    def gan_output_path(self) -> str:
        return (f'{self.output_path}/'
                f'CcGAN_{self.gan_arch}_{self.threshold_type}'
                f'_si{self.kernel_sigma:.3f}_ka{self.kappa:.3f}'
                f'_{self.loss_type}_nDs{self.num_d_steps}'
                f'_nDa{self.num_grad_acc_d}_nGa{self.num_grad_acc_g}'
                f'_Dbs{self.batch_size_d}_Gbs{self.batch_size_g}_{self.version}')

    @property
    def niqe_dump_path(self) -> str:  # NIQE 计算的伪造数据存储路径
        # return f"{self.output_path}/NIQE_{self.img_size}x{self.img_size}"
        return f"{self.gan_output_path}/saved_images/fake_images"

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
        pretty += f"gan_output_path: {self.gan_output_path}\n"
        pretty += f"niqe_dump_path: {self.niqe_dump_path}"
        return pretty

    ## 单例模式
    ## 法一,使用__new__实现
    # _instance = None
    #
    # def __new__(cls, *args, **kwargs):
    #     if cls._instance is None:
    #         cls._instance = super(ConfigModel, cls).__new__(cls)
    #     return cls._instance
    ### 上面这个实现会在使用时报错, 原因是__new__ 破坏了 Pydantic 的 BaseModel
    ### Pydantic 的 BaseModel 依赖 __init__ 进行字段初始化，但 __new__ 只创建对象，并不会调用 __init__
    ### 这导致 Pydantic 认为 ConfigModel 没有任何字段，访问 version 就会报错

    # 法二,使用@classmethod实现单例模式
    ## 为什么不命名为__instance? 因为如果私密属性,python对其执行“名称重整"为_ConfigModel__instance
    ## 这样子类就不能直接访问 __instance，如果你想扩展 ConfigModel，会很麻烦!
    _instance: ClassVar["BaseConfig"] = None  # 确保 `_instance` 不是字段

    @classmethod
    def instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()  # 确保单例
        return cls._instance

    @classmethod
    def from_yaml(cls, file_path: str):
        """从 YAML 加载配置，覆盖默认值"""
        import yaml
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


# 直接创建默认配置（单例）
cfg = BaseConfig.instance()
