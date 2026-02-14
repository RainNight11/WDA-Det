import os
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.total_steps = 0
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

    def save_networks(self, save_filename):
        save_path = os.path.join(self.save_dir, save_filename)
        os.makedirs(self.save_dir, exist_ok=True)
        tmp_path = f"{save_path}.tmp.{os.getpid()}"

        # serialize model and optimizer to dict
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'total_steps' : self.total_steps,
        }

        try:
            try:
                # Fast path: default zipfile serialization.
                torch.save(state_dict, tmp_path)
            except RuntimeError as exc:
                msg = str(exc)
                # Some filesystems intermittently fail with zip writer errors.
                if ("PytorchStreamWriter failed writing file" in msg) or ("unexpected pos" in msg):
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    # Retry with legacy serialization for better compatibility.
                    torch.save(
                        state_dict,
                        tmp_path,
                        _use_new_zipfile_serialization=False,
                    )
                else:
                    raise

            # Atomic replace: avoid half-written checkpoints.
            os.replace(tmp_path, save_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


    def eval(self):
        self.model.eval()

    def test(self):
        with torch.no_grad():
            self.forward()


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
