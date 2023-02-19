##
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets


##트레이닝 파라미터 설정

lr = 1e-3   # learning rate
batch_size = 4
num_epoch = 100

data_dir = "./datasets"
ckpt_dir = "./checkpoint"   # train 된 network 가 저장되는 checkpoint
log_dir = "./log"           # tensorboard log files
result_dir = "./result"     # 어떤 result ?

mode = "train"
train_continue = "off"

##
# select, cpu or gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##

