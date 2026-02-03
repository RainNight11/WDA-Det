class BaseConfig:
    """Base configuration with common parameters."""
    # Model & training mode
    mode = "binary"
    arch = "RFNT"
    fix_backbone = True

    # Data
    data_mode = "wang2020"
    wang2020_data_path = "../Datasets/"
    real_list_path = None
    fake_list_path = None
    data_label = "train"
    loadSize = 256
    cropSize = 224
    resize_or_crop = "scale_and_crop"
    no_flip = False

    # Data Augmentation
    rz_interp = "bilinear"
    blur_prob = 0.5
    blur_sig = "0.0,3.0"
    jpg_prob = 0.5
    jpg_method = "cv2,pil"
    jpg_qual = "30,100"

    # Training
    batch_size = 24
    num_threads = 24
    niter = 50
    lr = 0.0005
    beta1 = 0.9
    optim = "adam"
    weight_decay = 0.0

    # Training control
    gpu_ids = "0"
    name = "default_exp"
    checkpoints_dir = "./checkpoints"
    earlystop_epoch = 5
    loss_freq = 400
    save_epoch_freq = 1
    epoch_count = 1
    train_split = "train"
    val_split = "val"
    last_epoch = 0
    lastload_path = None

    # Flags
    data_aug = False
    attack_aug = False
    new_optim = False
    class_bal = False
    serial_batches = False
    init_type = "normal"
    init_gain = 0.02
    suffix = ""
