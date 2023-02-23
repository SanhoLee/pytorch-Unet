##
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from dataset import *
from model import UNet


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

## 네트워크 학습하기

transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

# Train 데이터 셋과 데이터 로더
dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)    # DataLoader는 표준모듈임

# Validation 데이서 셋과 데이터 로더
dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
loader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)


## 네트워크 생성
net = UNet().to(device)

## 손실함수 정의
fn_loss = nn.BCEWithLogitsLoss().to(device)

## Optimizer 정의
optim = torch.optim.Adam(net.parameters(), lr=lr)

## 그 외 부수적인 variables 정의하기.
num_data_train = len(dataset_train)
num_data_val = len(dataset_val)

# 한 배치당 사용되는 데이터 갯수를 정의
num_batch_train = np.ceil(num_data_train / batch_size)
num_batch_val = np.ceil(num_data_val / batch_size)

## Output 저장하기 위함.

# tensor to Numpy // tensor 를 numpy 형태로 바꾸기 위해서는 GPU 에 올려지 있는 data를 cpu로 복사해야 하는데 이 작업을 여기서 한다.
# transpose 를 사용해서, 인덱스 자리도 다시 수정해준다. 여기서 4개의 차원을 가진걸로 보여지는데, 맨 첫번째 차원은 무엇을 의미하는지 의문이다.
fn_toNumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)
# de-normalization
fn_denorm = lambda x, mean, std: (x*std) + mean
# classification, 네트워크 아웃풋을 바이너리 클래스 분류해주는 function, threshold > 0.5
fn_class = lambda x: 1.0 * (x > 0.5)

## Tensorboard 를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

## Training Network.

st_epoch = 0 # training 이 시작되는 epoch 을 설정



