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

lr = 1e-3  # learning rate
batch_size = 4
num_epoch = 100

data_dir = "./datasets"
ckpt_dir = "./checkpoint"  # train 된 network 가 저장되는 checkpoint
log_dir = "./log"  # tensorboard log files
result_dir = "./result"  # 어떤 result ?

mode = "train"
train_continue = "off"

##
# select, cpu or gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## 네트워크 학습하기

transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

# Train 데이터 셋과 데이터 로더
dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)  # DataLoader는 표준모듈임
# loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)  # for local machine...

# Validation 데이서 셋과 데이터 로더
dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
loader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

# 네트워크 생성
net = UNet().to(device)

# 손실함수 정의
fn_loss = nn.BCEWithLogitsLoss().to(device)

# Optimizer 정의
optim = torch.optim.Adam(net.parameters(), lr=lr)

# 그 외 부수적인 variables 정의하기.
num_data_train = len(dataset_train)
num_data_val = len(dataset_val)

# 한 배치당 사용되는 데이터 갯수를 정의
num_batch_train = np.ceil(num_data_train / batch_size)
num_batch_val = np.ceil(num_data_val / batch_size)

# Output 저장하기 위함.

# tensor to Numpy // tensor 를 numpy 형태로 바꾸기 위해서는 GPU 에 올려지 있는 data를 cpu로 복사해야 하는데 이 작업을 여기서 한다.
# transpose 를 사용해서, 인덱스 자리도 다시 수정해준다. 여기서 4개의 차원을 가진걸로 보여지는데, 맨 첫번째 차원은 무엇을 의미하는지 의문이다.
fn_toNumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
# de-normalization
fn_denorm = lambda x, mean, std: (x * std) + mean
# classification, 네트워크 아웃풋을 바이너리 클래스 분류해주는 function, threshold > 0.5
fn_class = lambda x: 1.0 * (x > 0.5)

# Tensorboard 를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))


## Save Network
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save(
        {
            'net': net.state_dict(),
            'optim': optim.state_dict()
        },
        "./%s/model_epoch%d.pth" % (ckpt_dir, epoch))


## Load Network
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))   # 가장 마지막 요소를 가져옴

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch

## Training and Validation of Network.
st_epoch = 0  # training 이 시작되는 epoch 을 설정

# 저장한 네트워크 있으면, 불러와서 train 진행함
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net ,optim=optim)
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
