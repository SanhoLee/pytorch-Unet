## import packages
import os
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data


## Dataloader
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        # data의 transform이 있을 경우에는 데이터 적용한다
        self.data_dir = data_dir
        self.transform = transform

        # 한 dir 안에 input, label 데이터가 같이 있기 때문에, 두 변수로 나눠준다.
        lst_data = os.listdir(self.data_dir)
        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        # add list variables
        self.lst_label = lst_label
        self.lst_input = lst_input

##
    def __len__(self):
        return len(self.lst_label)

##
    def __getitem__(self, index):
        # 인덱스 선택으로 해당 데이터를 가져올 수 있는 built-in method.

        # np array(npy) 파일을 불러오기
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # make array value into 0 to 1.
        label = label / 255.0
        input = input / 255.0

##      어레이 차원 확인 후, 3차원 매트릭스로 변경, channel 데이터 인덱스 자리를 만들기 위해서 ?
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

##      label, input 데이터를 사전형으로 준비
        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)
            # transform 클래스의 return 값은 여기서 선언한 data 사전형과 동일하게 해줘야 한다.

        return data


## Trnasform class
# class ToTensor(object):




