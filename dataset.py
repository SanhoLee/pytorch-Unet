## import packages
import os

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.utils.data
from torchvision import transforms

data_dir = "./datasets"
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

        lst_label.sort()
        lst_input.sort()

        # add list variables
        self.lst_label = lst_label
        self.lst_input = lst_input

#  Data length, method ocf // len()
    def __len__(self):
        return len(self.lst_label)

#  Get a specific index data, method of // instanceName(index)
    def __getitem__(self, index):
        # 인덱스 선택으로 해당 데이터를 가져올 수 있는 built-in method.

        # np array(npy) 파일을 불러오기
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # make array value into 0 to 1.
        label = label / 255.0
        input = input / 255.0

#      어레이 차원 확인 후, 3차원 매트릭스로 변경, channel 데이터 인덱스 자리를 만들기 위해서 ?
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

#      label, input 데이터를 사전형으로 준비
        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)
            # transform 클래스의 return 값은 여기서 선언한 data 사전형과 동일하게 해줘야 한다.

        return data




## Transform classes, It will be used when creating transform combination, likes 'transforms.Compose([...])'
class ToTensor(object):
    '''
    chagne data type : numpy to tensor
    '''
    def __call__(self, data):
        label, input = data['label'], data['input']

        # numpy to tensor, change and set to tensor-manner in terms of its index position.
        label = label.transpose((2,0,1)).astype(np.float32)
        input = input.transpose((2,0,1)).astype(np.float32)

        # map to return data
        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input) }

        return data

class Normalization(object):
    '''
    Nomarlizing pixel values
    '''

    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std
    def __call__(self, data):
        # Only applying to input data,
        # label data is consist of only 0 and 1(vary with 2 values),
        # binary data, don't need to normalization.

        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}
        return data

class RandomFlip(object):
    '''
    Literally, Flip data Left and Right, Up and Down Randomly
    '''
    def __call__(self, data):
        label, input = data['label'], data['input']

        # 난수 생성에 의한 데이터 조작
        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np. flipud(input)

        data = {'label': label, 'input': input}
        return data



## Test Dataset class with Transform classes
transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
data = dataset_train.__getitem__(3)

input = data['input']
label = data['label']

##
plt.subplot(121)
plt.title('input_withTransform')
plt.imshow(input.squeeze())     # 마지막 채널 인덱스를 없애준다.

plt.subplot(122)
plt.title('label_withTransform')
plt.imshow(label.squeeze())     # 마지막 채널 인덱스를 없애준다.

plt.show()