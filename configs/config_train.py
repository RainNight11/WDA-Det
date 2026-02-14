import os

from configs.base_config import BaseConfig
from options.train_options import TrainOptions


SUPPORTED_WAVELETS = ("db4", "coif2", "bior4.4", "sym8")
DEFAULT_WAVELET = "db4"
DEFAULT_WAVELET_LEVELS = 3
DEFAULT_WAVELET_THETA_INIT = 0.02


def _parse_env_bool(key):
    raw = os.getenv(key, "").strip()
    if raw == "":
        return None
    raw = raw.lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean for {key}: {raw}")


def _parse_env_float(key):
    raw = os.getenv(key, "").strip()
    if raw == "":
        return None
    return float(raw)


def _parse_env_int(key):
    raw = os.getenv(key, "").strip()
    if raw == "":
        return None
    return int(raw)

EXPERIMENT_NAME = os.getenv(
    "ICME_EXPERIMENT_NAME",
    "wda_consistency_v1_WildRF",
)   # 选择想要train的实验组（新模型可切到 wda_decision_fusion_v1_WildRF）
EXPERIMENT_WAVELET_NAME = os.getenv("ICME_WAVELET_NAME", "").strip()
EXPERIMENT_NAME_SUFFIX = os.getenv("ICME_EXPERIMENT_SUFFIX", "").strip()
EXPERIMENT_ABLATION_TAG = os.getenv("ICME_ABLATION_TAG", "").strip()
EXPERIMENT_DENOISE_MODE = os.getenv("ICME_DENOISE_MODE", "").strip()
EXPERIMENT_USE_RESIDUAL = _parse_env_bool("ICME_USE_RESIDUAL")
EXPERIMENT_EVIDENCE_POOL_MODE = os.getenv("ICME_EVIDENCE_POOL_MODE", "").strip()
EXPERIMENT_GATE_MODE = os.getenv("ICME_GATE_MODE", "").strip()
EXPERIMENT_AUX_ACTIVATION = os.getenv("ICME_AUX_ACTIVATION", "").strip()
EXPERIMENT_USE_EVIDENCE_BRANCH = _parse_env_bool("ICME_USE_EVIDENCE_BRANCH")
EXPERIMENT_SUPERVISED_LAMBDA = _parse_env_float("ICME_SUPERVISED_LAMBDA")
EXPERIMENT_CONSISTENCY_LAMBDA = _parse_env_float("ICME_CONSISTENCY_LAMBDA")
EXPERIMENT_NITER = _parse_env_int("ICME_NITER")
########################################
# ConfigTrain：默认参数 + 实验配置覆盖
########################################
class ConfigTrain(BaseConfig):
    def __init__(self):
        # Step 1: 先用 BaseOptions + TrainOptions 解析命令行/默认值
        parsed = TrainOptions().parse(print_options=False)
        args = vars(parsed)

        # 把这些默认参数都先挂到 self 上
        for k, v in args.items():
            setattr(self, k, v)

        # Step 2: 从预设实验字典中读取当前实验配置
        if EXPERIMENT_NAME not in EXPERIMENT_CONFIGS:
            raise ValueError(
                f"Unknown experiment preset: {EXPERIMENT_NAME}. "
                f"Available: {list(EXPERIMENT_CONFIGS.keys())}"
            )

        exp_cfg = EXPERIMENT_CONFIGS[EXPERIMENT_NAME]

        # 用实验配置覆盖默认参数
        for k, v in exp_cfg.items():
            setattr(self, k, v)

        if not getattr(self, "wavelet_name", ""):
            self.wavelet_name = DEFAULT_WAVELET
        if not getattr(self, "wavelet_levels", None):
            self.wavelet_levels = DEFAULT_WAVELET_LEVELS
        if not getattr(self, "wavelet_theta_init", None):
            self.wavelet_theta_init = DEFAULT_WAVELET_THETA_INIT

        if EXPERIMENT_WAVELET_NAME:
            if EXPERIMENT_WAVELET_NAME not in SUPPORTED_WAVELETS:
                raise ValueError(
                    f"Unsupported wavelet preset: {EXPERIMENT_WAVELET_NAME}. "
                    f"Supported: {list(SUPPORTED_WAVELETS)}"
                )
            self.wavelet_name = EXPERIMENT_WAVELET_NAME

        if EXPERIMENT_NAME_SUFFIX:
            self.name = f"{self.name}_{EXPERIMENT_NAME_SUFFIX}"
        if EXPERIMENT_ABLATION_TAG:
            self.ablation_tag = EXPERIMENT_ABLATION_TAG
        if EXPERIMENT_DENOISE_MODE:
            self.denoise_mode = EXPERIMENT_DENOISE_MODE
        if EXPERIMENT_USE_RESIDUAL is not None:
            self.use_residual = EXPERIMENT_USE_RESIDUAL
        if EXPERIMENT_EVIDENCE_POOL_MODE:
            self.evidence_pool_mode = EXPERIMENT_EVIDENCE_POOL_MODE
        if EXPERIMENT_GATE_MODE:
            self.gate_mode = EXPERIMENT_GATE_MODE
        if EXPERIMENT_AUX_ACTIVATION:
            self.aux_activation = EXPERIMENT_AUX_ACTIVATION
        if EXPERIMENT_USE_EVIDENCE_BRANCH is not None:
            self.use_evidence_branch = EXPERIMENT_USE_EVIDENCE_BRANCH
        if EXPERIMENT_SUPERVISED_LAMBDA is not None:
            self.supervised_lambda = EXPERIMENT_SUPERVISED_LAMBDA
        if EXPERIMENT_CONSISTENCY_LAMBDA is not None:
            self.consistency_lambda = EXPERIMENT_CONSISTENCY_LAMBDA
        if EXPERIMENT_NITER is not None:
            self.niter = EXPERIMENT_NITER

        # Step 3: 一些自动补全逻辑（可选）
        if not getattr(self, "checkpoints_dir", ""):
            self.checkpoints_dir = "checkpoints"

        # 如果没有手动给 lastload_path，就自动按 name 拼一个
        if not getattr(self, "lastload_path", ""):
            self.lastload_path = os.path.join(
                self.checkpoints_dir,
                self.name,
                "model_epoch_best.pth"
            )

        self.checkpoints_dir = os.path.join("checkpoints",self.data_name)  # 代码会自动拼接 + self.name
        # self.checkpoints_dir = os.path.join("checkpoints/FullProgan/")  # 代码会自动拼接，checkpoints/name
        self.lastload_path = os.path.join(self.checkpoints_dir, self.name, "model_epoch_best.pth")


# 2. 定义所有实验的配置字典
########################################
EXPERIMENT_CONFIGS = {
"wda_consistency_v1_WildRF": dict(
        name="wda_consistency_v1_WildRF",  # 实验名（保存目录用）
        arch="RFNT-CLIP:ViT-L/14",  # WDA consistency model
        loss="loss_bce",
        lr=0.001,
        niter=5,
        gpu_ids="0",
        batch_size=32,
        num_threads=8,

        data_mode="WildRF",
        data_name="WildRF",
        # TODO: 更新为你的数据路径
        wang2020_data_path="/hy-tmp/WildRF/",
        last_epoch=-1,

        fix_backbone=True,
        data_aug=True,
        jpg_prob=0.5,
        jpg_qual=[30, 100],
        blur_prob=0.3,
        blur_sig=[0.0, 2.0],
        consistency_lambda=0.1,
        consistency_warmup=0.1,
        consistency_ema_decay=0.99,
        consistency_noise_std=0.01,
        consistency_blur_prob=0.5,
        consistency_blur_sigma_min=0.5,
        consistency_blur_sigma_max=1.2,
        consistency_resize_scale=0.1,
        wavelet_name="db4",
        wavelet_levels=3,
        wavelet_theta_init=0.02,
        wavelet_freeze_epochs=0,
        learn_wavelet=False,
    ),
"wda_consistency_v1_fdmas": dict(
        name="wda_consistency_v1_fdmas",  # 实验名（保存目录用）
        arch="RFNT-CLIP:ViT-L/14",  # WDA consistency model
        loss="loss_bce",
        lr=0.001,
        niter=5,
        gpu_ids="0",
        batch_size=32,
        num_threads=8,

        data_mode="wang2020",
        data_name="Progan",
        # TODO: 更新为你的数据路径
        wang2020_data_path="/hy-tmp/WildRF/",
        last_epoch=-1,

        fix_backbone=True,
        data_aug=True,
        jpg_prob=0.5,
        jpg_qual=[30, 100],
        blur_prob=0.3,
        blur_sig=[0.0, 2.0],
        consistency_lambda=0.1,
        consistency_warmup=0.1,
        consistency_ema_decay=0.99,
        consistency_noise_std=0.01,
        consistency_blur_prob=0.5,
        consistency_blur_sigma_min=0.5,
        consistency_blur_sigma_max=1.2,
        consistency_resize_scale=0.1,
        wavelet_name="db4",
        wavelet_levels=3,
        wavelet_theta_init=0.02,
        wavelet_freeze_epochs=0,
        learn_wavelet=False,
    ),
"wda_decision_fusion_v1_WildRF": dict(
        name="wda_decision_fusion_v1_WildRF",  # 实验名（保存目录用）
        arch="RFNTDF-CLIP:ViT-L/14",  # WDA decision-fusion model
        loss="loss_bce",
        lr=0.001,
        niter=5,
        gpu_ids="0",
        batch_size=32,
        num_threads=8,

        data_mode="WildRF",
        data_name="WildRF",
        # TODO: 更新为你的数据路径
        wang2020_data_path="/hy-tmp/WildRF/",
        last_epoch=-1,

        fix_backbone=True,
        data_aug=True,
        jpg_prob=0.5,
        jpg_qual=[30, 100],
        blur_prob=0.3,
        blur_sig=[0.0, 2.0],
        consistency_lambda=0.1,
        consistency_warmup=0.1,
        consistency_ema_decay=0.99,
        consistency_noise_std=0.01,
        consistency_blur_prob=0.5,
        consistency_blur_sigma_min=0.5,
        consistency_blur_sigma_max=1.2,
        consistency_resize_scale=0.1,
        wavelet_name="db4",
        wavelet_levels=3,
        wavelet_theta_init=0.02,
        wavelet_freeze_epochs=0,
        learn_wavelet=False,
        ablation_tag="A0_full",
        denoise_mode="wavelet",
        use_evidence_branch=True,
        use_residual=True,
        evidence_pool_mode="aligned",
        gate_mode="learned",
        aux_activation="tanh",
        supervised_lambda=1.0,
    ),
"wda_decision_fusion_v1_fdmas": dict(
        name="wda_decision_fusion_v1_fdmas",  # 实验名（保存目录用）
        arch="RFNTDF-CLIP:ViT-L/14",  # WDA decision-fusion model
        loss="loss_bce",
        lr=0.001,
        niter=5,
        gpu_ids="0",
        batch_size=32,
        num_threads=8,

        data_mode="wang2020",
        data_name="Progan",
        # TODO: 更新为你的数据路径
        wang2020_data_path="/hy-tmp/WildRF/",
        last_epoch=-1,

        fix_backbone=True,
        data_aug=True,
        jpg_prob=0.5,
        jpg_qual=[30, 100],
        blur_prob=0.3,
        blur_sig=[0.0, 2.0],
        consistency_lambda=0.1,
        consistency_warmup=0.1,
        consistency_ema_decay=0.99,
        consistency_noise_std=0.01,
        consistency_blur_prob=0.5,
        consistency_blur_sigma_min=0.5,
        consistency_blur_sigma_max=1.2,
        consistency_resize_scale=0.1,
        wavelet_name="db4",
        wavelet_levels=3,
        wavelet_theta_init=0.02,
        wavelet_freeze_epochs=0,
        learn_wavelet=False,
        ablation_tag="A0_full",
        denoise_mode="wavelet",
        use_evidence_branch=True,
        use_residual=True,
        evidence_pool_mode="aligned",
        gate_mode="learned",
        aux_activation="tanh",
        supervised_lambda=1.0,
    ),
    
"image-denoised-clip": dict(
        name="image-denoised-clip",  # 实验名（保存目录用）
        arch="RFNT-CLIP:ViT-L/14",  # backbone
        # 'CLIP:ViT-B/32',
        # 'CLIP:ViT-B/16',
        # 'CLIP:ViT-L/14',
        # 'CLIP:RN50',
        # 'CLIP:RN101'
        # 'RFNT-CLIP:RN50',  # CLIP:RN50
        # 'RFNT-CLIP:ViT-L/14',  # 
        #  RFNT-DINOv2:ViT-L14
        loss="loss_bce",  # loss_t:采用复合损失 loss_total; loss_bce; loss_real_consistency
        lr=0.0005,
        niter=5,  # 轮数
        gpu_ids="0",
        batch_size=32,
        num_threads=24,

        # Progan
        # data_mode="wang2020",
        # data_name="Progan",  # WildRF
        # wang2020_data_path="../Datasets/",

        data_mode="wang2020",  # WildRF ,wang2020
        data_name="Progan",  # WildRF/PartProgan/Progan
        #wang2020_data_path="../../dataset/CNNSpot/progan4cls",
        #wang2020_data_path="/data_B/tianyu/dataset/WildRF/",
        wang2020_data_path="/data1/liaoty/CNNSpot/progan4cls/progan4cls",
        # 关于训练轮数，如果是从头训练，last_epoch 设 -1；
        # 如果是接着之前的训练继续训，last_epoch 设上次训练的最后 epoch 编号
        last_epoch= -1,
        # checkpoints_dir="checkpoints/PartProgan/RFNTCLIP",

        fix_backbone=True,
        jpg_prob=0.0,
        jpg_qual=[50, 100],
        blur_prob=0.0,
        blur_sig=[0.0, 3.0],
    ),

    "image_denoised-attention-resnet": dict(
        name="image_denoised-attention-resnet",  # 实验名（保存目录用）
        arch="RFNT-CLIP:ViT-L/14",  # backbone
        # 'CLIP:ViT-B/32',
        # 'CLIP:ViT-B/16',
        # 'CLIP:ViT-L/14',
        # 'CLIP:RN50',
        # 'CLIP:RN101'
        # 'RFNT-CLIP:RN50',  # CLIP:RN50
        # 'RFNT-CLIP:ViT-L/14',  # 
        #  RFNT-DINOv2:ViT-L14
        loss="loss_bce",  # loss_t:采用复合损失 loss_total; loss_bce; loss_real_consistency
        lr=0.0005,
        niter=30,  # 轮数
        gpu_ids="1",
        batch_size=32,
        num_threads=24,

        # Progan
        # data_mode="wang2020",
        # data_name="Progan",  # WildRF
        # wang2020_data_path="../Datasets/",

        data_mode="wang2020",  # WildRF ,wang2020
        data_name="PartProgan",  # WildRF/PartProgan
        wang2020_data_path="../../dataset/CNNSpot/progan4cls",
        #wang2020_data_path="/data_B/tianyu/dataset/WildRF/",
        # 关于训练轮数，如果是从头训练，last_epoch 设 -1；
        # 如果是接着之前的训练继续训，last_epoch 设上次训练的最后 epoch 编号
        last_epoch= -1,
        # checkpoints_dir="checkpoints/PartProgan/RFNTCLIP",

        fix_backbone=True,
        jpg_prob=0.0,
        jpg_qual=[50, 100],
        blur_prob=0.0,
        blur_sig=[0.0, 3.0],
    ),
    "image_denoised-attention-resnet-for-wildrf": dict(
        name="image_denoised-attention-resnet-for-wildrf",  # 实验名（保存目录用）
        arch="RFNT-CLIP:ViT-L/14",  # backbone
        # 'CLIP:ViT-B/32',
        # 'CLIP:ViT-B/16',
        # 'CLIP:ViT-L/14',
        # 'CLIP:RN50',
        # 'CLIP:RN101'
        # 'RFNT-CLIP:RN50',  # CLIP:RN50
        # 'RFNT-CLIP:ViT-L/14',  # 
        #  RFNT-DINOv2:ViT-L14
        loss="loss_bce",  # loss_t:采用复合损失 loss_total; loss_bce; loss_real_consistency
        lr=0.0005,
        niter=30,  # 轮数
        gpu_ids="0",
        batch_size=32,
        num_threads=24,

        # Progan
        # data_mode="wang2020",
        # data_name="Progan",  # WildRF
        # wang2020_data_path="../Datasets/",

        data_mode="WildRF",  # WildRF ,wang2020
        data_name="WildRF",  # WildRF/PartProgan
        # wang2020_data_path="../../dataset/CNNSpot/progan4cls",
        wang2020_data_path="/data_B/tianyu/dataset/WildRF/",
        # 关于训练轮数，如果是从头训练，last_epoch 设 -1；
        # 如果是接着之前的训练继续训，last_epoch 设上次训练的最后 epoch 编号
        last_epoch= -1,
        # checkpoints_dir="checkpoints/PartProgan/RFNTCLIP",

        fix_backbone=True,
        jpg_prob=0.0,
        jpg_qual=[50, 100],
        blur_prob=0.0,
        blur_sig=[0.0, 3.0],
    ),
    # dinov2
    "image_denoised-attention-for-wildrf": dict(
        name="image_denoised-attention-for-wildrf",  # 实验名（保存目录用）
        arch="RFNT-CLIP:ViT-L/14",  # backbone
        # 'CLIP:ViT-B/32',
        # 'CLIP:ViT-B/16',
        # 'CLIP:ViT-L/14',
        # 'CLIP:RN50',
        # 'CLIP:RN101'
        # 'RFNT-CLIP:RN50',  # CLIP:RN50
        # 'RFNT-CLIP:ViT-L/14',  # 
        #  RFNT-DINOv2:ViT-L14
        loss="loss_bce",  # loss_t:采用复合损失 loss_total; loss_bce; loss_real_consistency
        lr=0.0005,
        niter=30,  # 轮数
        gpu_ids="0",
        batch_size=32,
        num_threads=24,

        # Progan
        # data_mode="wang2020",
        # data_name="Progan",  # WildRF
        # wang2020_data_path="../Datasets/",

        data_mode="WildRF",  # WildRF ,wang2020
        data_name="WildRF",  # WildRF/PartProgan
        # wang2020_data_path="../../dataset/CNNSpot/progan4cls",
        wang2020_data_path="/data_B/tianyu/dataset/WildRF/",
        # 关于训练轮数，如果是从头训练，last_epoch 设 -1；
        # 如果是接着之前的训练继续训，last_epoch 设上次训练的最后 epoch 编号
        last_epoch= 14,
        # checkpoints_dir="checkpoints/PartProgan/RFNTCLIP",

        fix_backbone=True,
        jpg_prob=0.0,
        jpg_qual=[50, 100],
        blur_prob=0.0,
        blur_sig=[0.0, 3.0],
    ),

    # 对损失函数的消融实验
    "rfnt_vitl14_loss_r_diag_corr_noisenet": dict(
        name="rfnt_vitl14_loss_r_diag_corr_noisenet",  # 实验名（保存目录用）
        arch="RFNT-CLIP:ViT-L/14",  # backbone
        loss="loss_real_consistency",  # loss_t:采用复合损失 loss_total; loss_bce; loss_real_consistency
        lr=0.0005,
        niter=50,
        gpu_ids="0",
        batch_size=32,
        num_threads=24,

        # Progan-train
        # data_mode="wang2020",
        # data_name="Progan",  # WildRF
        # wang2020_data_path="../Datasets/",

        # WildRF
        data_mode="WildRF",  # WildRF
        data_name="WildRF",  # WildRF/PartProgan
        wang2020_data_path="../../dataset/WildRF",

        # 关于训练轮数，如果是从头训练，last_epoch 设 -1；
        # 如果是接着之前的训练继续训，last_epoch 设上次训练的最后 epoch 编号
        last_epoch=-1,
        # checkpoints_dir="checkpoints/PartProgan/RFNTCLIP",

        fix_backbone=True,
        jpg_prob=0.0,
        jpg_qual=[50, 100],
        blur_prob=0.0,
        blur_sig=[0.0, 3.0],
    ),

    # 实验 1：RFNT: ViT-L/14, example
    "rfnt_vit_l14_loss_t": dict(
        name="rfnt_vitl14_losst_dca_noisenet",   # 实验名（保存目录用）
        arch="RFNT-CLIP:ViT-L/14",                    # backbone
        loss="loss_t",                           # loss_t:采用复合损失 loss_total; loss_bce; loss_real_consistency
        lr=0.0005,
        niter=50,
        gpu_ids="0",
        batch_size=32,
        num_threads=24,

        # WildRF
        # data_mode="WildRF",  # WildRF
        # data_name="WildRF",  # WildRF/PartProgan
        # wang2020_data_path="../../dataset/WildRF",  # tianyu_nyx

        # Progan
        data_mode="wang2020",
        data_name="Progan",  # WildRF
        wang2020_data_path="../Datasets/",

        # data_mode="wang2020",
        # data_name = "Progan",   # WildRF
        # wang2020_data_path="../Datasets/progan_1percent",

        # 关于训练轮数，如果是从头训练，last_epoch 设 -1；
        # 如果是接着之前的训练继续训，last_epoch 设上次训练的最后 epoch 编号
        last_epoch= -1,
        # checkpoints_dir="checkpoints/PartProgan/RFNTCLIP",

        fix_backbone=True,
        jpg_prob=0.0,
        jpg_qual=[50, 100],
        blur_prob=0.0,
        blur_sig=[0.0, 3.0],
    ),

    "rfnt_vitl14_losst_diag_corr_noisenet": dict(
        name="rfnt_vitl14_losst_dca_noisenet",   # 实验名（保存目录用）
        arch="RFNT-CLIP:ViT-L/14",                    # backbone
        loss="loss_t",                           # loss_t:采用复合损失 loss_total; loss_bce:
        lr=0.0005,
        niter=50,
        gpu_ids="0",
        batch_size=32,
        num_threads=24,

        data_mode="wang2020",
        data_name = "Progan",   # WildRF
        wang2020_data_path="../Datasets/",

        # 关于训练轮数，如果是从头训练，last_epoch 设 -1；
        # 如果是接着之前的训练继续训，last_epoch 设上次训练的最后 epoch 编号
        last_epoch= -1,
        # checkpoints_dir="checkpoints/PartProgan/RFNTCLIP",

        fix_backbone=True,
        jpg_prob=0.0,
        jpg_qual=[50, 100],
        blur_prob=0.0,
        blur_sig=[0.0, 3.0],
    ),

    # 实验 2：ViT-L/14 + loss_t + ProGAN 1%
    "rfnt_vit_l14_loss_t_npr": dict(
        name="rfnt_vitl14_losst_dca_noisenet_npr",   # 实验名（保存目录用）
        arch="RFNT-CLIP:ViT-L/14",                    # backbone
        loss="loss_t",                           # loss_t:采用复合损失 loss_total; loss_bce:
        lr=0.0005,
        niter=50,
        gpu_ids="1",
        batch_size=32,
        num_threads=24,

        data_mode="wang2020",
        data_name = "Progan",   # WildRF
        # wang2020_data_path="../Datasets/progan_1percent",
        wang2020_data_path="../Datasets/",

        # 关于训练轮数，如果是从头训练，last_epoch 设 -1；
        # 如果是接着之前的训练继续训，last_epoch 设上次训练的最后 epoch 编号
        last_epoch=3,
        # checkpoints_dir="checkpoints/PartProgan/RFNTCLIP",
        fix_backbone=True,
        jpg_prob=0.0,
        jpg_qual=[50, 100],
        blur_prob=0.0,
        blur_sig=[0.0, 3.0],
    ),

    # 实验 2：DINOv2 + BCE + Full ProGAN
    "dinov2_bce": dict(
        name="rfnt_dinov2_l14_bcel_feat_gauss0_05",
        arch="RFNT:DINOv2-ViT-L14",
        loss="loss_bce",
        lr=1e-5,
        niter=50,
        gpu_ids="0",
        batch_size=32,
        num_threads=24,

        data_mode="wang2020",
        wang2020_data_path="../Datasets/",

        last_epoch=-1,
        checkpoints_dir="checkpoints/FullProgan",

        fix_backbone=True,
        jpg_prob=0.0,
        jpg_qual=[50, 100],
        blur_prob=0.0,
        blur_sig=[0.0, 3.0],
    ),
    # 后面你可以继续加：
    # "vit_l14_loss_bce": dict(...),
    # "vit_l14_noise05": dict(...),
}



# parser.add_argument('--rz_interp', default='bilinear')
# parser.add_argument('--blur_prob', type=float, default=0.5)
# parser.add_argument('--blur_sig', default='0.0,3.0')
# parser.add_argument('--jpg_prob', type=float, default=0.5)
# parser.add_argument('--jpg_method', default='cv2,pil')
# parser.add_argument('--jpg_qual', default='30,100')

# 'CLIP:ViT-B/32',
# 'CLIP:ViT-B/16',
# 'CLIP:ViT-L/14',
# 'CLIP:RN50',
# 'CLIP:RN101'
# 'RFNT:RN50',
# 'RFNT-CLIP:ViT-L/14',

# """DINOv2"""
# # Step 2: 再定义自定义字段（优先级更高）
# self.name = "rfnt_dinov2_l14_bcel_feat_gauss0.05"  # **********
# self.arch = "RFNT:DINOv2-ViT-L14"  # **********
# '''
# 'RFNT:RN50',
# 'RFNT-CLIP:ViT-L/14',
# 'RFNT:DINOv2-ViT-L14',
# '''
# self.loss = "loss_bce"  # loss_t:采用复合损失 loss_total; loss_bce:
# self.lr = 0.00001  # 第一轮0.0005 第二轮开始0.00001
# self.niter = 50
# self.gpu_ids = "0"
# self.batch_size = 32
# self.num_threads = 24
#
# self.data_mode = "wang2020"
# self.wang2020_data_path = "../Datasets/"  # 完整数据集
# # nyx服务器
# # self.wang2020_data_path = "../../dataset/CNNSpot/progan_1percent"  # 部分数据集
# # self.wang2020_data_path = "../Datasets/progan_1percent"  # 部分数据集
# self.last_epoch = -1
# self.checkpoints_dir = os.path.join("checkpoints/FullProgan/")  # 代码会自动拼接，checkpoints/name
# self.lastload_path = os.path.join(self.checkpoints_dir, self.name, "model_epoch_best.pth")
#
# # self.bk_name = "RN50"  # backbone class
