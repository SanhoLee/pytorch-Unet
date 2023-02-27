##
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from model import UNet
from dataset import *
from util import load

##트레이닝 파라미터 설정

lr = 1e-3  # learning rate
batch_size = 4
num_epoch = 100

data_dir = "./datasets"
ckpt_dir = "./checkpoint"  # train 된 network 가 저장되는 checkpoint
log_dir = "./log"  # tensorboard log files
result_dir = "./result"  # test 결과를 저장

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

mode = "train"
train_continue = "off"

##
# select, cpu or gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## 네트워크 학습하기
# test 에서는 RandomFlip을 삭제함.

transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

# Testing data
dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)  # DataLoader는 표준모듈임

# 네트워크 생성
net = UNet().to(device)

# 손실함수 정의
fn_loss = nn.BCEWithLogitsLoss().to(device)

# Optimizer 정의
optim = torch.optim.Adam(net.parameters(), lr=lr)

# 그 외 부수적인 variables, function 정의하기.
num_data_test = len(dataset_test)
num_batch_test = np.ceil(num_data_test / batch_size)

fn_toNumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)


## Training and Validation of Network.
# st_epoch = 0  # training 이 시작되는 epoch 을 설정

# 저장한 네트워크 있으면, 불러와서 train 진행함
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

'''
    evaluation 에서는, net, optim 정보만 활용한다.
    st_epoch의 경우는 train 하는 경우에 필요한 변수이며, 여기서는 사용되고 있지 않다.
    
'''

# ------------------------------------------------------------------------------------
# Validation of Network.
# no back propagation step, in order to avoid this, torch.no_grad() is needed.

with torch.no_grad():  # disabled gradient calculation.
    net.eval()  # network 에게 validation 임을 명시해주는 부분
    loss_arr = []

    for batch, data in enumerate(loader_test, 1):
        # forward pass
        label = data['label'].to(device)  # 왜 validation 일때는 to(device)를 쓰는거지?
        input = data['input'].to(device)

        output = net(input)

        # skip backward propagation step because it is validation of network.

        # 손실함수 계산하기
        loss = fn_loss(output, label)
        loss_arr += [loss.item()]

        print("TEST : BATCH %04d / %04d | LOSS %.4f" %
              (batch, num_batch_test, np.mean(loss_arr)))

        # Tensorborad 에 input, output, label 을 저장
        label = fn_toNumpy(label)
        input = fn_toNumpy(fn_denorm(input, mean=0.5, std=0.5))
        output = fn_toNumpy(fn_class(output))  # output 데이터는 이진화 함수를 적용시켜 0, 1 데이터로 분류해준다.

        # png, numpy 타입 각각 형태로 결과를 저장한다.
        for j in range(label.shape[0]):     # label.shape : NWHC 형태이며, N 은 한 배치 데이터 갯수를 나타냄
            id = num_batch_test * (batch - 1) + j

            plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

            np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

# 마지막 출력. 전체 평균 loss 값
print("AVERAGE TEST : BATCH %04d / %04d | LOSS %.4f" % (batch, num_batch_test, np.mean(loss_arr)))





