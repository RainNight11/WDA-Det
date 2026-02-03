from .clip_models import CLIPModel
from .imagenet_models import ImagenetModel
from .rfnt_models import RFNTModel
from .moe_models import MoEDDModel


VALID_NAMES = [
    'Imagenet:resnet18',
    'Imagenet:resnet34',
    'Imagenet:resnet50',
    'Imagenet:resnet101',
    'Imagenet:resnet152',
    'Imagenet:vgg11',
    'Imagenet:vgg19',
    'Imagenet:swin-b',
    'Imagenet:swin-s',
    'Imagenet:swin-t',
    'Imagenet:vit_b_16',
    'Imagenet:vit_b_32',
    'Imagenet:vit_l_16',
    'Imagenet:vit_l_32',

    'CLIP:RN50', 
    'CLIP:RN101', 
    'CLIP:RN50x4', 
    'CLIP:RN50x16', 
    'CLIP:RN50x64', 
    'CLIP:ViT-B/32', 
    'CLIP:ViT-B/16', 
    'CLIP:ViT-L/14', 
    'CLIP:ViT-L/14@336px',

    # 以下用的全是CLIP: 版本的
    'RFNT-CLIP:RN50',
    'RFNT-CLIP:ViT-L/14',
    'RFNT-DINOv2:ViT-G14',
    'RFNT-DINOv2:ViT-L14',  # 目前使用
    'RFNT-DINOv2:ViT-B14',
    'RFNT-DINOv2:ViT-S14',

    'MoEDD'  # 混合专家检测Diffusion生成图像
]





def get_model(name):
    assert name in VALID_NAMES
    if name.startswith("Imagenet:"):
        return ImagenetModel(name[9:]) 
    elif name.startswith("CLIP:"):
        return CLIPModel(name[5:])
    elif name.startswith("RFNT"):
        return RFNTModel(name[5:])
    elif name.startswith("MoEDD"):
        return MoEDDModel("MoEDD")
    else:
        assert False 
