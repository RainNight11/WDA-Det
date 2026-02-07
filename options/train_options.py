from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--earlystop_epoch', type=int, default=5)
        parser.add_argument('--data_aug', action='store_true', help='if specified, perform additional data augmentation (photometric, blurring, jpegging)')
        parser.add_argument('--attack_aug', action='store_true', help='if specified, perform additional data augmentation (attack aug)')
        parser.add_argument('--optim', type=str, default='adam', help='optim to use [sgd, adam]')
        # for resuming train
        parser.add_argument('--new_optim', action='store_true', help='new optimizer instead of loading the optim state')
        parser.add_argument('--last_epoch', type=int, default=-1, help='starting epoch count for scheduler intialization')
        parser.add_argument('--lastload_path',type=str,default='',help='path to last checkpoint')
        parser.add_argument('--loss_freq', type=int, default=400, help='frequency of showing loss on tensorboard')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--train_split', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--val_split', type=str, default='val', help='train, val, test, etc')
        parser.add_argument('--niter', type=int, default=15, help='total epoches')  # epoch
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--consistency_lambda', type=float, default=0.0, help='weight for consistency loss; 0 disables')
        parser.add_argument('--consistency_warmup', type=float, default=0.1, help='warmup ratio of total steps')
        parser.add_argument('--consistency_noise_std', type=float, default=0.01, help='max noise std for consistency aug')
        parser.add_argument('--consistency_blur_prob', type=float, default=0.5, help='probability to apply blur in consistency aug')
        parser.add_argument('--consistency_blur_sigma_min', type=float, default=0.5, help='min gaussian sigma for blur aug')
        parser.add_argument('--consistency_blur_sigma_max', type=float, default=1.2, help='max gaussian sigma for blur aug')
        parser.add_argument('--consistency_resize_scale', type=float, default=0.1, help='resize scale range for consistency aug')

        self.isTrain = True
        return parser
