##
import argparse

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
from util import *

## Parser 생성하기

# object 생성
parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")
parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")


# Parsing argument what program takes from user by command line.
args = parser.parse_args()

##트레이닝 파라미터 설정

lr = args.lr  # learning rate
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir  # train 된 network 가 저장되는 checkpoint
log_dir = args.log_dir  # tensorboard log files
result_dir = args.result_dir  # 어떤 result ?
mode = args.mode
train_continue = args.train_continue

## 디렉토리 생성하기
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

##
# select, cpu or gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## print arguments

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("data dir: %s" % data_dir)
print("checkpoint dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("result dir: %s" % result_dir)
print("mode: %s" % mode)
print("train continue: %s" % train_continue)



## 네트워크 학습하기

if mode == 'train':
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

    # Train 데이터 셋과 데이터 로더
    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)  # DataLoader는 표준모듈임

    # Validation 데이서 셋과 데이터 로더
    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
    loader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

    # 그 외 부수적인 variables 정의하기.
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    # 한 배치당 사용되는 데이터 갯수를 정의
    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)

else:
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

    # Testing data
    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
    loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False,
                             num_workers=0)  # DataLoader는 표준모듈임
    # 그 외 부수적인 variables, function 정의하기.
    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)


## 네트워크 생성
net = UNet().to(device)

# 손실함수 정의
fn_loss = nn.BCEWithLogitsLoss().to(device)

# Optimizer 정의
optim = torch.optim.Adam(net.parameters(), lr=lr)

# Output 저장하기 위함.
fn_toNumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

# Tensorboard 를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

## Training and Validation of Network.
st_epoch = 0


# TRAIN mode
if mode == 'train':
    if train_continue == 'on':
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    for epoch in range(st_epoch + 1, num_epoch + 1):
        # ------------------------------------------------------------------------------------
        # Training of Network.
        net.train()  # network에게 train 세션이라는 것을 명시함
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):  # 최초 1 부터 enumerate counting 한다.
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)    # cpu? gpu? 에 따라서, data도 그 형태에 맞게 설정해줘야 한다?


            output = net(input)

            # backward pass
            optim.zero_grad()

            loss = fn_loss(output, label)  # 네트워크를 통과한 output과, 이미 알고있는 label 데이터를 가지고 loss 를 추정한다?
            loss.backward()

            optim.step()

            # 손실함수 계산
            loss_arr += [loss.item()]
    ##
            print("TRAIN : EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            # Tensorborad 에 input, output, label 을 저장

            # 저장해야 되는 input, output, label 데이터는 모두 tensor 형태이기 때문에
            # toNumpy 메소드를 최종적으로 적용시켜 데이터를 numpy 형태로 변경한다.
            label = fn_toNumpy(label)
            input = fn_toNumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_toNumpy(fn_class(output))  # output 데이터는 이진화 함수를 적용시켜 0, 1 데이터로 분류해준다.

            writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            '''
            NHWC :
                N - Number of images in the batch
                H - height of the image
                W - width of the image
                C - Number of Channels of the image
            
            '''

        # loss 를 tensorboard에 저장
        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        # ------------------------------------------------------------------------------------
        # Validation of Network.
        # no back propagation step, in order to avoid this, torch.no_grad() is needed.

        with torch.no_grad():  # disabled gradient calculation.
            net.eval()  # network 에게 validation 임을 명시해주는 부분
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                # forward pass
                label = data['label'].to(device)  # 왜 validation 일때는 to(device)를 쓰는거지?
                input = data['input'].to(device)

                output = net(input)

                # skip backward propagation step because it is validation of network.

                # 손실함수 계산하기
                loss = fn_loss(output, label)
                loss_arr += [loss.item()]

                print("VALID : EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

                # Tensorborad 에 input, output, label 을 저장
                label = fn_toNumpy(label)
                input = fn_toNumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_toNumpy(fn_class(output))  # output 데이터는 이진화 함수를 적용시켜 0, 1 데이터로 분류해준다.

                writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        # Save Network with every epoch proceeding
        if epoch % 50 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)


    # close two writers
    writer_train.close()
    writer_val.close()

# TEST mode
else:
    # evaluation의 경우는 저장된 네트워크를 반드시 불러온다.
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

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
            for j in range(label.shape[0]):  # label.shape : NWHC 형태이며, N 은 한 배치 데이터 갯수를 나타냄
                id = num_batch_test * (batch - 1) + j

                plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

                np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

    # 마지막 출력. 전체 평균 loss 값
    print("AVERAGE TEST : BATCH %04d / %04d | LOSS %.4f" % (batch, num_batch_test, np.mean(loss_arr)))





