# validate.py（或你定义 ValConfig 的文件）
import os

########################################
# 1. 选择当前要用的验证配置
########################################
VAL_EXPERIMENT_NAME = "wda_consistency_v1_wildRF"   # 在这里切换：wildrf_vitl14 / fdmas_dinov2 等


########################################
# 2. 定义所有验证实验的配置字典
########################################
VAL_EXPERIMENT_CONFIGS = {
    "wda_consistency_v1_WildRF": dict(
        arch="RFNT-CLIP:ViT-L/14",
        result_folder="TestResults/WildRF",
        batch_size=64,
        gpu_ids="0",
        # 加载哪个 checkpoint 来测试
        checkpoint_dir="checkpoints/WildRF/wda_consistency_v1_WildRF",
        max_sample=None,
        jpeg_quality=None,
        gaussian_sigma=None,
        disable_ssl_verify=False,

        # 测试用的数据根目录
        dataroot="/hy-tmp/WildRF/test/",
        data_mode="RFNT_WildRF",
        # 测试哪个 epoch：可以是 '2' 或 '7-10'
        val_epoch="5",
    ),
    "wda_consistency_v1_fdmas": dict(
        arch="RFNT-CLIP:ViT-L/14",
        result_folder="TestResults/fdmas",
        batch_size=64,
        gpu_ids="0",
        # 加载哪个 checkpoint 来测试
        checkpoint_dir="checkpoints/WildRF/wda_consistency_v1_fdmas",
        max_sample=None,
        jpeg_quality=None,
        gaussian_sigma=None,
        disable_ssl_verify=False,

        # 测试用的数据根目录
        dataroot="/hy-tmp/WildRF/test/",
        data_mode="RFNT_WildRF",
        # 测试哪个 epoch：可以是 '2' 或 '7-10'
        val_epoch="5",
    ),
     # 
    "image-denoised-clip": dict(
        arch="RFNT-CLIP:ViT-L/14",
        result_folder="TestResults/Progan4cls_train/fdmas/", # 才修改好
        batch_size=64,
        gpu_ids="5",
        # 加载哪个 checkpoint 来测试
        checkpoint_dir="checkpoints/Progan/image-denoised-clip", 
        max_sample=None,
        jpeg_quality=None,
        gaussian_sigma=None,
        disable_ssl_verify=False,

        # 测试用的数据根目录
        dataroot="/data1/liaoty/fdmas/test",  # fdmas
        data_mode="RFNT_fdmas",
        # 测试哪个 epoch：可以是 '2' 或 '7-10'
        val_epoch="3-4",
    ),

        # 
    "image_denoised-attention-resnet-for-wildrf": dict(
        arch="RFNT-CLIP:ViT-L/14",
        result_folder="TestResults/WildRF", # 才修改好
        batch_size=128,
        gpu_ids="5",
        # 加载哪个 checkpoint 来测试
        checkpoint_dir="checkpoints/WildRF/image_denoised-attention-resnet-for-wildrf", 
        max_sample=None,
        jpeg_quality=None,
        gaussian_sigma=None,
        disable_ssl_verify=False,

        # 测试用的数据根目录
        dataroot="/data_B/tianyu/dataset/WildRF/test/",  # fdmas
        data_mode="RFNT_WildRF",
        # 测试哪个 epoch：可以是 '2' 或 '7-10'
        val_epoch="0-26",
    ),
        # 
    "image_denoised-attention-for-wildrf": dict(
        arch="RFNT-CLIP:ViT-L/14",
        result_folder="TestResults/WildRF", # 才修改好
        batch_size=128,
        gpu_ids="5",
        # 加载哪个 checkpoint 来测试
        checkpoint_dir="checkpoints/WildRF/image_denoised-attention-for-wildrf", 
        max_sample=None,
        jpeg_quality=None,
        gaussian_sigma=None,
        disable_ssl_verify=False,

        # 测试用的数据根目录
        dataroot="/data_B/tianyu/dataset/WildRF/test/",  # fdmas
        data_mode="RFNT_WildRF",
        # 测试哪个 epoch：可以是 '2' 或 '7-10'
        val_epoch="3-10",
    ),
    # 

        # 
    "image_denoised-mlp-for-wildrf": dict(
        arch="RFNT-CLIP:ViT-L/14",
        result_folder="TestResults/image_denoised-mlp-for-wildrf", # 才修改好
        batch_size=128,
        gpu_ids="5",
        # 加载哪个 checkpoint 来测试
        checkpoint_dir="checkpoints/WildRF/image_denoised-mlp-for-wildrf", 
        max_sample=None,
        jpeg_quality=None,
        gaussian_sigma=None,
        disable_ssl_verify=False,

        # 测试用的数据根目录
        dataroot="/data_B/tianyu/dataset/WildRF/test/",  # fdmas
        data_mode="RFNT_WildRF",
        # 测试哪个 epoch：可以是 '2' 或 '7-10'
        val_epoch="4-14",
    ),
    # 
    "rfnt_dinov2_loss_r_diag_corr_noisenet_normal_0.5": dict(
        arch="RFNT-DINOv2:ViT-L14",
        result_folder="TestResults/Progan_train/fdmas/", # 才修改好
        batch_size=128,
        gpu_ids="1",
        # 加载哪个 checkpoint 来测试
        checkpoint_dir="checkpoints/Progan/rfnt_dinov2_loss_r_diag_corr_noisenet", 
        max_sample=None,
        jpeg_quality=None,
        gaussian_sigma=None,
        disable_ssl_verify=False,

        # 测试用的数据根目录
        dataroot="../Datasets/fdmas/test",  # fdmas
        data_mode="RFNT_fdmas",
        # 测试哪个 epoch：可以是 '2' 或 '7-10'
        val_epoch="5-6",
    ),

    # 配置 1：在 WildRF 上验证 ViT-L/14 的模型
    "rfnt_vitl14_loss_r_diag_corr_noisenet": dict(
        arch="RFNT:ViT-L/14",
        result_folder="TestResults/Progan_train/fdmas/",
        batch_size=128,
        gpu_ids="1",
        # 加载哪个 checkpoint 来测试
        checkpoint_dir="checkpoints/Progan/rfnt_vitl14_loss_r_diag_corr_noisenet",
        max_sample=None,
        jpeg_quality=None,
        gaussian_sigma=None,
        disable_ssl_verify=False,

        # 测试用的数据根目录
        dataroot="../Datasets/fdmas/test",  # fdmas
        data_mode="RFNT_fdmas",
        # 测试哪个 epoch：可以是 '2' 或 '7-10'
        val_epoch="9-12",
    ),
    # 配置 1：在 WildRF 上验证 ViT-L/14 的模型
    "wildrf_vitl14": dict(
        arch="RFNT:ViT-L/14",
        result_folder="TestResults/WildRF/RFNTCLIP/",
        data_mode="RFNT_WildRF",  # 需要修改
        batch_size=128,
        gpu_ids="1",
        # 加载哪个 checkpoint 来测试
        checkpoint_dir="checkpoints/PartProgan/RFNTCLIP/",

        ablation_group="ba1",
        max_sample=None,
        jpeg_quality=None,
        gaussian_sigma=None,
        disable_ssl_verify=False,

        # 测试用的数据根目录
        dataroot="../Datasets/WildRF/test",
        # 测试哪个 epoch：可以是 '2' 或 '7-10'
        val_epoch="2",
    ),

    # 配置 1：在 WildRF 上验证 ViT-L/14 的模型
    "rfnt_vit_l14_loss_t_npr": dict(
        arch="RFNT:ViT-L/14",
        #result_folder="TestResults/PartProgan_train/fdmas_sample_test/rfnt_vitl14_losst_dca_noisenet_npr",
        result_folder="TestResults/Progan_train/fdmas/",
        # data_mode="RFNT_fdmas",
        batch_size=128,
        gpu_ids="0",
        # 加载哪个 checkpoint 来测试
        checkpoint_dir="checkpoints/Progan/rfnt_vitl14_losst_dca_noisenet_npr",

        ablation_group="ba1",
        max_sample=None,
        jpeg_quality=None,
        gaussian_sigma=None,
        disable_ssl_verify=False,

        # 测试用的数据根目录
        # dataroot="../Datasets/fdmas_sample",
        # dataroot="../Datasets/Chameleon/test",
        # data_mode="RFNT_Chameleon",
        # 测试哪个 epoch：可以是 '2' 或 '7-10'
        dataroot="../Datasets/fdmas/test",  # fdmas
        data_mode="RFNT_fdmas",
        val_epoch="best",
    ),
    # 11/26 checkpoints/Progan/rfnt_vitl14_losst_diag_corr_noisenet ： Train Progan/ Test fdmas
    "rfnt_vitl14_losst_diag_corr_noisenet": dict(
        arch="RFNT:ViT-L/14",
        #result_folder="TestResults/PartProgan_train/fdmas_sample_test/rfnt_vitl14_losst_dca_noisenet_npr",
        result_folder="TestResults/Progan_train/fdmas/rfnt_vitl14_losst_diag_corr_noisenet",
        # data_mode="RFNT_fdmas",
        batch_size=128,
        gpu_ids="1",
        # 加载哪个 checkpoint 来测试
        checkpoint_dir="checkpoints/Progan/rfnt_vitl14_losst_diag_corr_noisenet",

        # ablation_group="ba1",
        max_sample=None,
        jpeg_quality=None,
        gaussian_sigma=None,
        disable_ssl_verify=False,

        # 测试用的数据根目录
        # dataroot="../Datasets/fdmas_sample",
        # dataroot="../Datasets/Chameleon/test",
        # data_mode="RFNT_Chameleon",
        # 测试哪个 epoch：可以是 '2' 或 '7-10'
        dataroot="../Datasets/fdmas/test",  # fdmas
        data_mode="RFNT_fdmas",
        val_epoch="13",  # 这里的n指的是checkpoint保存的第n-1轮
    ),

    "rfnt_vitl14_losst_dca_noisenet": dict(
        arch="RFNT:ViT-L/14",
        #result_folder="TestResults/PartProgan_train/fdmas_sample_test/rfnt_vitl14_losst_dca_noisenet_npr",
        result_folder="TestResults/Progan_train/fdmas/rfnt_vitl14_losst_dca_noisenet",
        # data_mode="RFNT_fdmas",
        batch_size=128,
        gpu_ids="1",
        # 加载哪个 checkpoint 来测试
        checkpoint_dir="checkpoints/Progan/rfnt_vitl14_losst_dca_noisenet",

        # ablation_group="ba1",
        max_sample=None,
        jpeg_quality=None,
        gaussian_sigma=None,
        disable_ssl_verify=False,

        # 测试用的数据根目录
        # dataroot="../Datasets/fdmas_sample",
        # dataroot="../Datasets/Chameleon/test",
        # data_mode="RFNT_Chameleon",
        # 测试哪个 epoch：可以是 '2' 或 '7-10'
        dataroot="../Datasets/fdmas/test",  # fdmas
        data_mode="RFNT_fdmas",
        val_epoch="0-5",  # 这里的n指的是checkpoint保存的第n-1轮
    ),

    #
    "rfnt_clip_vit14_best": dict(
        arch="RFNT:ViT-L/14",
        # result_folder="TestResults/PartProgan_train/fdmas_sample_test/rfnt_vitl14_losst_dca_noisenet_npr",
        result_folder="TestResults/Progan_train/fdmas/rfnt_clip_vitl14_best",
        # data_mode="RFNT_fdmas",
        batch_size=128,
        gpu_ids="1",
        # 加载哪个 checkpoint 来测试
        checkpoint_dir="checkpoints/Progan/rfnt_clip_vitl14_best",

        # ablation_group="ba1",
        max_sample=None,
        jpeg_quality=None,
        gaussian_sigma=None,
        disable_ssl_verify=False,

        # 测试用的数据根目录
        # dataroot="../Datasets/fdmas_sample",
        # dataroot="../Datasets/Chameleon/test",
        # data_mode="RFNT_Chameleon",
        # 测试哪个 epoch：可以是 '2' 或 '7-10'
        dataroot="../Datasets/fdmas/test",  # fdmas
        data_mode="RFNT_fdmas",
        val_epoch="9",  # 这里的n指的是checkpoint保存的第n-1轮
    ),

    # 配置 2：在 fdmas 上验证 DINOv2 模型
    "fdmas_dinov2": dict(
        arch="RFNT:DINOv2-ViT-L14",
        result_folder="TestResults/Progan_train/Chamelon/rfnt_dinov2_l14_feat_gauss0.05",
        batch_size=128,
        gpu_ids="0",
        checkpoint_dir="checkpoints/FullProgan/rfnt_dinov2_l14_feat_gauss0.05",

        ablation_group="ba1",
        max_sample=None,
        jpeg_quality=None,
        gaussian_sigma=None,
        disable_ssl_verify=False,

        dataroot="../Datasets/fdmas_sample",
        val_epoch="7-10",
    ),
    # 以后你可以继续加：
    # "wildrf_dinov2": dict(...),
    # "fdmas_vitl14": dict(...),
}


########################################
# 3. ValConfig：根据 VAL_EXPERIMENT_NAME 选择配置
########################################
class ValConfig:
    """验证 / 测试阶段配置（支持多实验预设）"""
    def __init__(self):
        # 检查当前选择的实验名是否存在
        if VAL_EXPERIMENT_NAME not in VAL_EXPERIMENT_CONFIGS:
            raise ValueError(
                f"Unknown val experiment preset: {VAL_EXPERIMENT_NAME}. "
                f"Available: {list(VAL_EXPERIMENT_CONFIGS.keys())}"
            )

        cfg = VAL_EXPERIMENT_CONFIGS[VAL_EXPERIMENT_NAME]

        # 把字典里的配置全部挂到 self 上
        for k, v in cfg.items():
            setattr(self, k, v)

        self.result_folder = os.path.join(self.result_folder,VAL_EXPERIMENT_NAME)
        # 这里你也可以加一些自动补全逻辑，比如：
        # 确保 checkpoint_dir 是一个存在的目录（可选）
        # if not os.path.exists(self.checkpoint_dir):
        #     print(f"[Warn] checkpoint_dir does not exist: {self.checkpoint_dir}")
