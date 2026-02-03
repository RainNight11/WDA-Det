from torchvision import transforms
from utils import *

from datetime import datetime


class setting_config:
    """
    the config of training setting.
    """

    network = 'vmunet'
    model_config = {
        'num_classes': 1,
        'input_channels': 3,
        # ----- VM-UNet ----- #
        'depths': [2, 2, 2, 2],
        'depths_decoder': [2, 2, 2, 1],
        'drop_path_rate': 0.2,
        'load_ckpt_path': './pre_trained_weights/vmamba_small_e238_ema.pth',
    }

    datasets = 'UltraEdit'
    if datasets == 'isic18':
        data_path = './data/isic2018/'
    elif datasets == 'isic17':
        data_path = './data/isic2017/'
    elif datasets == 'UltraEdit':
        data_path = "../../datasets/UltraEdit"
    else:
        raise Exception('datasets in not right!')

    criterion = BceDiceLoss(wb=1, wd=1)

    pretrained_path = './pre_trained/'
    num_classes = 1
    input_size_h = 256
    input_size_w = 256
    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = 16
    seed = 42
    world_size = None
    rank = None
    amp = False
    gpu_id = '0'
    batch_size = 4
    epochs = 300

    work_dir = 'results/' + network + '_' + datasets + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    print_interval = 20
    val_interval = 30
    save_interval = 100
    threshold = 0.5
    only_test_and_save_figs = False
    best_ckpt_path = 'results/vmunet_UltraEdit_Thursday_22_May_2025_17h_16m_01s/checkpoints/best.pth'
    img_save_path = 'test_images/'

