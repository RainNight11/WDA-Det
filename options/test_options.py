from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.set_defaults(arch='RFNTDF-CLIP:ViT-L/14')
         #parser.add_argument('--premodel_path',default="./pretrained_weights/fc_weights.pth")  # our 直接将res50用作默认预训练模型
        parser.add_argument('--premodel_path', default="./pretrained_weights/model_epoch_best.pth")
        parser.add_argument('--wavelet_name', type=str, default='db4', help='wavelet basis: db4/coif2/bior4.4/sym8')
        parser.add_argument('--wavelet_levels', type=int, default=3, help='wavelet decomposition levels')
        parser.add_argument('--wavelet_theta_init', type=float, default=0.02, help='initial shrinkage theta for learnable wavelet')
        parser.add_argument('--learn_wavelet', action='store_true', help='enable learnable wavelet shrinkage')
        parser.add_argument('--no_resize', action='store_true')
        parser.add_argument('--no_crop', action='store_true')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--predict_path', type=str, default='predict', help='')
        parser.add_argument('--test_dataset_path', type=str, default='./datasets/faceB', help='')

        self.isTrain = False
        return parser

