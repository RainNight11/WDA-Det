import os
import pickle
import sys
from copy import deepcopy
from io import BytesIO
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random
from datetime import datetime
from sklearn.metrics import average_precision_score, accuracy_score
import csv
import torch.utils.data
import sys
from dataset_paths import DATASET_PATHS
from data.datasets import data_augment, custom_resize, rz_dict
from models import get_model
from configs.config_validate import ValConfig
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import logging
from datetime import datetime
import time



SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711]
}

# some functions
def find_best_threshold(y_true, y_pred):
    """
    We assume first half is real(0), second half is fake(1).
    Return the threshold that yields the best accuracy.
    """
    N = y_true.shape[0]
    if y_pred[0:N//2].max() <= y_pred[N//2:N].min():
        return (y_pred[0:N//2].max() + y_pred[N//2:N].min()) / 2

    best_acc = 0
    best_thres = 0
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp >= thres] = 1
        temp[temp < thres] = 0
        acc = (temp == y_true).sum() / N
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc
    return best_thres

def png2jpg(img, quality):
    """
    将PNG图像转为JPEG，quality用于控制JPEG的画质（压缩率）
    """
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality)
    img = Image.open(out)
    img = np.array(img)
    out.close()
    return Image.fromarray(img)

def gaussian_blur(img, sigma):
    """
    对图像做高斯模糊
    """
    arr = np.array(img)
    gaussian_filter(arr[:, :, 0], output=arr[:, :, 0], sigma=sigma)
    gaussian_filter(arr[:, :, 1], output=arr[:, :, 1], sigma=sigma)
    gaussian_filter(arr[:, :, 2], output=arr[:, :, 2], sigma=sigma)
    return Image.fromarray(arr)

def calculate_acc(y_true, y_pred, thres):
    """
    在给定threshold时，分别计算真实集和假集的准确率，以及总体准确率
    """
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > thres)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc

#######其实datasets.PY 里面有
def recursively_read(rootdir, must_contain, exts=["png","PNG","jpg", "JPG","JPEG", "jpeg"]):
    out = []
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out

def get_list(path, must_contain=''):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [ item for item in image_list if must_contain in item   ]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list

class RealFakeDataset(Dataset):
    def __init__(self,
                 data_mode,
                 dataset_name,
                 dataset_class,
                 max_sample,
                 arch,
                 dataroot,
                 jpeg_quality=None,
                 gaussian_sigma=None,
                 opt=None):
        
        assert data_mode in ["wang2020", "RFNT","Plot"]
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma
        if dataset_name == 'fdmas' or dataset_name == 'WildRF':
            root_dir = dataroot   # r'../Datasets/fdmas_sample'
            real_list = get_list(os.path.join(root_dir, dataset_class), must_contain='0_real')
            fake_list = get_list(os.path.join(root_dir, dataset_class), must_contain='1_fake')
        elif dataset_name == 'Chameleon':
            root_dir = dataroot   # r'../Datasets/fdmas_sample'
            real_list = get_list(os.path.join(root_dir), must_contain='0_real')
            fake_list = get_list(os.path.join(root_dir), must_contain='1_fake')
        elif data_mode == 'Plot':
            root_dir = r'datasets/'
            real_list = get_list(os.path.join(root_dir,'samples'), must_contain='real')  # 用于绘图
            fake_list = get_list(os.path.join(root_dir, 'samples'), must_contain='fake')


        # if isinstance(real_path, str) and isinstance(fake_path, str):
        #     real_list, fake_list = self.read_path(real_path, fake_path, data_mode)
        # else:
        #     real_list = []
        #     fake_list = []
        #     for rpath, fpath in zip(real_path, fake_path):
        #         rl, fl = self.read_path(rpath, fpath, data_mode)
        #         real_list += rl
        #         fake_list += fl

        self.total_list = real_list + fake_list
        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        if opt is None:
            stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
            self.transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from])
            ])
        else:
            self.transform = build_transform(opt, arch)


    def read_path(self, real_path, fake_path, data_mode, max_sample = None):
        if data_mode == 'wang2020':
            real_list = get_list(real_path, must_contain='0_real')
            fake_list = get_list(fake_path, must_contain='1_fake')
        else:
            real_list = get_list(real_path)
            fake_list = get_list(fake_path)

        if max_sample is not None:
            if (max_sample > len(real_list)) or (max_sample > len(fake_list)):
                max_sample = 1e9
                print("not enough images, max_sample falling to 100")
            random.shuffle(real_list)
            random.shuffle(fake_list)
            real_list = real_list[:max_sample]
            fake_list = fake_list[:max_sample]

        assert len(real_list) == len(fake_list), \
            f"real_list({len(real_list)}) vs fake_list({len(fake_list)}) length mismatch!"
        return real_list, fake_list

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]
        # img = Image.open(img_path).convert("RGB")
        try:
            img = Image.open(img_path).convert("RGB")
        except OSError:
            print(f"无法加载图片: {img_path}")  # 输出无法加载的图片路径
            return self.__getitem__((idx + 1) % len(self))  # 递归调用，跳过此图片并加载下一张



        if self.gaussian_sigma is not None:
            img = gaussian_blur(img, self.gaussian_sigma)
        if self.jpeg_quality is not None:
            img = png2jpg(img, self.jpeg_quality)

        img = self.transform(img)
        return img, label

def validate(model, loader, opt, find_thres=False):
    """
    对模型做验证。返回AP、在固定阈值=0.5下的准确率，以及最优阈值下的准确率等指标。
    """
    from sklearn.metrics import average_precision_score

    y_true, y_pred = [], []
    print("Length of dataset: %d" % (len(loader.dataset)))

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for img, label in tqdm(loader,desc = 'Validating',unit = 'batch'):
            in_tens = img.to(device)
            if opt.arch.startswith("RFNT"):
                output, _ = model(in_tens)
            elif opt.arch.startswith("CLIP:"):
                output = model(in_tens)
            scores = output.sigmoid().flatten().tolist()
            y_pred.extend(scores)
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ap = average_precision_score(y_true, y_pred)

    # 在阈值=0.5 的情况下计算准确率
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)

    if not find_thres:
        return ap, r_acc0, f_acc0, acc0

    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)
    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres


def build_transform(opt, arch):
    load_size = getattr(opt, "loadSize", 256)
    crop_size = getattr(opt, "cropSize", 224)

    if getattr(opt, "no_resize", False):
        rz_func = transforms.Lambda(lambda img: img)
    else:
        rz_opt = opt
        if not hasattr(opt, "loadSize") or not hasattr(opt, "rz_interp"):
            class _Tmp:  # lightweight shim
                pass
            rz_opt = _Tmp()
            rz_opt.loadSize = load_size
            rz_opt.rz_interp = getattr(opt, "rz_interp", ['bilinear'])
        rz_func = transforms.Lambda(lambda img: custom_resize(img, rz_opt))

    if getattr(opt, "no_crop", False):
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(crop_size)

    if getattr(opt, "isTrain", False) and not getattr(opt, "no_flip", False):
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)

    if getattr(opt, "isTrain", False) and getattr(opt, "data_aug", False):
        aug_func = transforms.Lambda(lambda img: data_augment(img, opt))
    else:
        aug_func = transforms.Lambda(lambda img: img)

    bk_name = arch[5:] if arch.startswith("RFNT") else arch
    stat_from = bk_name.split(":")[0].lower()

    return transforms.Compose([
        rz_func,
        aug_func,
        crop_func,
        flip_func,
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
    ])





def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}h {m:02d}m {s:02d}s"
    else:
        return f"{m:02d}m {s:02d}s"


class Validator:
    def __init__(self, config):
        self.opt = config

        # 日志文件路径（放在结果目录里）
        os.makedirs(self.opt.result_folder, exist_ok=True)
        log_path = os.path.join(self.opt.result_folder, "validate.log")

        # 日志器配置
        self.logger = logging.getLogger("Validator")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()  # 避免重复添加 handler

        # 文件 + 控制台双输出
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        os.environ["CUDA_VISIBLE_DEVICES"] = self.opt.gpu_ids
        self.device = torch.device(
            f"cuda:{self.opt.gpu_ids}" if torch.cuda.is_available() else "cpu"
        )

        # 记录启动时间和配置
        self.start_time = datetime.now()
        self.logger.info("=" * 80)
        self.logger.info("新一次验证开始")
        self.logger.info(f"启动时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"使用设备: {self.device}")
        self.logger.info("当前配置参数:")
        for k, v in vars(self.opt).items():
            self.logger.info(f"  {k}: {v}")

    def get_ckpt_epochs(self):
        self.logger.debug(f"checkpoint_dir = {self.opt.checkpoint_dir}")
        self.logger.debug(f"files in dir: {os.listdir(self.opt.checkpoint_dir)}")

        all_ckpts = sorted([
            f for f in os.listdir(self.opt.checkpoint_dir)
            if f.startswith("model_epoch_") and f.endswith(".pth")
        ])

        if "-" in self.opt.val_epoch:
            start, end = map(int, self.opt.val_epoch.split("-"))
            ckpts = [
                f"model_epoch_{i}.pth" for i in range(start, end + 1)
                if f"model_epoch_{i}.pth" in all_ckpts
            ]
        else:
            epoch_file = f"model_epoch_{int(self.opt.val_epoch)}.pth"
            ckpts = [epoch_file] if epoch_file in all_ckpts else []

        if not ckpts:
            self.logger.error(
                f"未在 {self.opt.checkpoint_dir} 中找到指定 epoch 的权重: {self.opt.val_epoch}"
            )
        else:
            self.logger.info(f"将要验证的 checkpoint: {ckpts}")
        return ckpts

    def validate_epoch(self, ckpt_file):
        epoch_num = ckpt_file.split("_")[-1].split(".")[0]
        ckpt_path = os.path.join(self.opt.checkpoint_dir, ckpt_file)
        result_path = os.path.join(self.opt.result_folder, f"epoch{epoch_num}")
        os.makedirs(result_path, exist_ok=True)

        self.logger.info(
            f"开始评估 checkpoint: {ckpt_path} -> 结果目录: {result_path}"
        )
        epoch_start = time.time()

        model = get_model(self.opt.arch)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict["model"])
        model.to(self.device).eval()

        epoch_results = []

        data_mode, dataset_name = self.opt.data_mode.split("_")

        # 针对 fdmas / WildRF 多子数据集的情况
        if dataset_name == "fdmas" or dataset_name == "WildRF":
            DATASET_sub = DATASET_PATHS[dataset_name]
            for dataset_path in DATASET_sub:
                set_seed()
                self.logger.info("*" * 61)
                self.logger.info(f"开始评估数据集: {dataset_path}")
                ds_start = time.time()

                dataset = RealFakeDataset(
                    data_mode=data_mode,
                    dataset_name=dataset_name,
                    dataset_class=dataset_path,
                    max_sample=self.opt.max_sample,
                    arch=self.opt.arch,
                    dataroot=self.opt.dataroot,
                    jpeg_quality=self.opt.jpeg_quality,
                    gaussian_sigma=self.opt.gaussian_sigma,
                    opt=self.opt,
                )
                loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.opt.batch_size,
                    shuffle=False,
                    num_workers=0,
                )

                results = validate(model, loader, self.opt, find_thres=True)
                ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres = results

                self.logger.info(
                    f"{dataset_path}  AP={ap:.4f}, "
                    f"Acc(0.5)={acc0:.4f}, Acc(best)={acc1:.4f}, "
                    f"best_thres={best_thres:.4f}"
                )

                ds_time = time.time() - ds_start
                self.logger.info(
                    f"数据集 {dataset_path} 评估耗时: {format_seconds(ds_time)}"
                )

                epoch_results.append({
                    "dataset": dataset_path,
                    "AP": round(ap * 100, 2),
                    "Acc(0.5)": round(acc0 * 100, 2),
                    "Acc(best)": round(acc1 * 100, 2),
                    "best_thres": round(best_thres, 4),
                    "r_acc0": round(r_acc0 * 100, 2),
                    "f_acc0": round(f_acc0 * 100, 2),
                    "r_acc1": round(r_acc1 * 100, 2),
                    "f_acc1": round(f_acc1 * 100, 2),
                })

            df = pd.DataFrame(epoch_results)
            result_csv_path = os.path.join(
                result_path, f"epoch_{epoch_num}_results.csv"
            )
            df.to_csv(result_csv_path, index=False)
            self.logger.info(
                f"epoch {epoch_num} 的结果已保存到 {result_csv_path}"
            )

        # 针对单一 Chameleon 数据集的情况
        elif dataset_name.startswith("Chameleon"):
            set_seed()
            self.logger.info("*" * 61)
            self.logger.info(f"开始评估数据集: {dataset_name}")
            ds_start = time.time()

            dataset = RealFakeDataset(
                data_mode=data_mode,
                dataset_name=dataset_name,
                dataset_class=dataset_name,
                max_sample=self.opt.max_sample,
                arch=self.opt.arch,
                dataroot=self.opt.dataroot,
                jpeg_quality=self.opt.jpeg_quality,
                gaussian_sigma=self.opt.gaussian_sigma,
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.opt.batch_size,
                shuffle=False,
                num_workers=0,
            )

            results = validate(model, loader, self.opt, find_thres=True)
            ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres = results

            self.logger.info(
                f"{dataset_name}  AP={ap:.4f}, "
                f"Acc(0.5)={acc0:.4f}, Acc(best)={acc1:.4f}, "
                f"best_thres={best_thres:.4f}"
            )

            ds_time = time.time() - ds_start
            self.logger.info(
                f"数据集 {dataset_name} 评估耗时: {format_seconds(ds_time)}"
            )

            epoch_results.append({
                "dataset": dataset_name,
                "AP": round(ap * 100, 2),
                "Acc(0.5)": round(acc0 * 100, 2),
                "Acc(best)": round(acc1 * 100, 2),
                "best_thres": round(best_thres, 4),
                "r_acc0": round(r_acc0 * 100, 2),
                "f_acc0": round(f_acc0 * 100, 2),
                "r_acc1": round(r_acc1 * 100, 2),
                "f_acc1": round(f_acc1 * 100, 2),
            })

            df = pd.DataFrame(epoch_results)
            result_csv_path = os.path.join(
                result_path, f"epoch_{epoch_num}_results.csv"
            )
            df.to_csv(result_csv_path, index=False)
            self.logger.info(
                f"epoch {epoch_num} 的结果已保存到 {result_csv_path}"
            )

        epoch_time = time.time() - epoch_start
        self.logger.info(
            f"checkpoint {ckpt_file} 整体评估耗时: {format_seconds(epoch_time)}"
        )

    def run(self):
        run_start = time.time()

        ckpt_files = self.get_ckpt_epochs()
        if not ckpt_files:
            self.logger.error(
                f"未找到可用 checkpoint，val_epoch = {self.opt.val_epoch}"
            )
            sys.exit(1)

        for ckpt_file in ckpt_files:
            self.validate_epoch(ckpt_file)

        total_time = time.time() - run_start
        self.logger.info(
            f"本次全部验证任务总耗时: {format_seconds(total_time)}"
        )
        self.logger.info("=" * 80)


if __name__ == "__main__":
    config = ValConfig()
    validator = Validator(config)
    validator.run()
