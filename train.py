import os
import time
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm
from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from configs.config_train import ConfigTrain  # 从配置文件导入
import copy
from datetime import datetime
from util import setup_logger
import torch


# 创建验证用配置
def get_val_opt(opt):
    val_opt = copy.deepcopy(opt)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = 'val'
    val_opt.jpg_method = ['pil']
    val_opt.arch = opt.arch
    val_opt.data_mode = opt.data_mode

    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]
    return val_opt


def count_parameters(model, trainable_only=False):
    """统计模型参数数量。

    Args:
        model: nn.Module
        trainable_only: True 时只统计 requires_grad=True 的参数
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    # === Step 1: 载入训练配置 ===
    opt = ConfigTrain()

    # === Step 3: 验证配置 ===
    val_opt = get_val_opt(opt)

    # === Step 4: 设置日志路径 ===
    log_file = os.path.join(opt.checkpoints_dir, opt.name, "training.log")
    train_logger = setup_logger("Train", log_file)

    # 设备 & 环境
    if hasattr(opt, "gpu_ids"):
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        device = torch.device(
            f"cuda:{opt.gpu_ids}" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 记录启动信息 & 配置
    start_time = datetime.now()
    train_logger.info("=" * 80)
    train_logger.info("新一次训练开始")
    train_logger.info(f"启动时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    train_logger.info(f"使用设备: {device}")
    train_logger.info("训练配置参数:")
    for k, v in vars(opt).items():
        train_logger.info(f"  {k}: {v}")
    train_logger.info("验证配置参数:")
    for k, v in vars(val_opt).items():
        train_logger.info(f"  {k}: {v}")

    # === Step 5: 模型和数据加载器 ===
    # === Step 5: 模型和数据加载器 ===
    model = Trainer(opt)

    # ---- 模型大小统计：完整模型 & 可训练参数 & backbone ----
    # Trainer 里通常封装了实际的 nn.Module，这里优先取 model.model，如果没有就直接用 model
    net = model.model if hasattr(model, 'model') else model

    # 完整模型参数量
    total_params = count_parameters(net, trainable_only=False)
    trainable_params = count_parameters(net, trainable_only=True)

    # 尝试获取 backbone（根据你自己网络定义做适配）
    backbone = net.pre_model
    if hasattr(net, 'pre_model'):
        backbone = net.pre_model
    elif hasattr(model, 'backbone'):
        backbone = model.backbone

    # backbone = net.pre_model
    if hasattr(net, 'pre_model'):
        backbone = net.pre_model
        backbone_total_params = count_parameters(backbone, trainable_only=False)
        backbone_trainable_params = count_parameters(backbone, trainable_only=True)
    else:
        backbone_total_params = 0
        backbone_trainable_params = 0
        print("Warning: 未找到 backbone (net.pre_model)")
        logging.warning("未找到 backbone (net.pre_model)")

    train_logger.info(f"Total params (all): {total_params} ({total_params / 1e6:.3f} M)")
    train_logger.info(f"Trainable params (all): {trainable_params} ({trainable_params / 1e6:.3f} M)")
    train_logger.info(f"Backbone total params: {backbone_total_params} ({backbone_total_params / 1e6:.3f} M)")
    train_logger.info(f"Backbone trainable params: {backbone_trainable_params} ({backbone_trainable_params / 1e6:.3f} M)")

    data_loader = create_dataloader(opt)
    val_loader = create_dataloader(val_opt)

    # === Step 6: TensorBoard 日志 ===
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    iter_start_time = time.time()

    train_logger.info("Length of data loader: %d", len(data_loader))

    start_epoch = (opt.last_epoch + 1) if opt.last_epoch != -1 else 0
    start_step = model.total_steps
    train_logger.info("Starting from epoch: %d", start_epoch)
    train_logger.info("Total epochs to run: %d", opt.niter)

    # === Step 7: 训练循环 ===
    for epoch in range(start_epoch, opt.niter):
        epoch_start_time = time.time()
        train_logger.info("-" * 80)
        train_logger.info("Epoch %d/%d 开始", epoch + 1, opt.niter)

        with tqdm(total=len(data_loader), desc=f"Epoch {epoch + 1}/{opt.niter}", unit="batch") as pbar:
            for i, data in enumerate(data_loader):
                model.total_steps += 1
                model.set_input(data)
                model.optimize_parameters()

                if model.total_steps % opt.loss_freq == 0:
                    avg_iter_time = (time.time() - iter_start_time) / max(1, model.total_steps)
                    train_logger.info(
                        "Train loss: %f at step: %d (avg iter time: %.4f s)",
                        model.loss, model.total_steps, avg_iter_time
                    )
                    train_writer.add_scalar('loss', model.loss, model.total_steps)

                pbar.update(1)

        if epoch % opt.save_epoch_freq == 0:
            train_logger.info('saving the model at the end of epoch %d', epoch)
            # model.save_networks('model_epoch_best.pth')
            model.save_networks('model_epoch_%s.pth' % epoch)

        # === Step 8: 验证 ===
        model.eval()
        ap, r_acc, f_acc, acc = validate(model.model, val_loader, opt)
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        train_logger.info("(Val @ epoch %d) acc: %f; ap: %f", epoch, acc, ap)

        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                train_logger.info("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                train_logger.info("Early stopping.")
                break

        model.train()
        epoch_duration = time.time() - epoch_start_time
        train_logger.info("Epoch %d duration: %.2f seconds", epoch + 1, epoch_duration)

    total_train_time = time.time() - iter_start_time
    train_logger.info("全部训练耗时: %.2f 秒 (%.2f 小时)",
                      total_train_time, total_train_time / 3600.0)
    train_logger.info("=" * 80)