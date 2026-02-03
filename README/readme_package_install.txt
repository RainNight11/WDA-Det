# CUDA 121 python 310
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

tensorboardX
tqdm
pandas
scikit-learn
scikit-image
ftfy
regex
matplotlib
opencv-python

# pip install tensorboardX tqdm pandas scikit-learn scikit-image ftfy regex matplotlib opencv-python

pytorch                   2.2.0           py3.10_cuda12.1_cudnn8.9.2_0    pytorch 
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install torchvision==0.17.0 pytorch-cuda=12.1 -c pytorch -c nvidia

cu121 py310

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install scikit-learn scikit-image ftfy regex opencv-python tensorboard tqdm pandas matplotlib

tensorboard --logdir checkpoints --port 6006

各个服务器上

"""
tianyu下：
dataroot="../../dataset/WildRF/test",

tianqi:
dataroot="../Datasets/WildRF/test",
"""

config_validate.py
        dataroot="../Datasets/fdmas/test",  # fdmas
        data_mode="RFNT_fdmas",

        