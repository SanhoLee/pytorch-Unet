# check train data and organizing.
## necessary pkg.
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

## load Data
dir_name = './datasets'
name_label = 'train-labels.tif'
name_input = 'train-volume.tif'

# 덩어리 채로 tif 파일을 읽어 들인 상태
img_label = Image.open(os.path.join(dir_name, name_label))
img_input = Image.open(os.path.join(dir_name, name_input))

# size of single img, number of frames it has.
ny, nx = img_label.size
nframe = img_label.n_frames

## set the number of train, validation and test data imgs
nframe_train = 24
nframe_val = 3
nframe_test = 3

## set some directories for saving.
dir_save_train = os.path.join(dir_name, 'train')
dir_save_val = os.path.join(dir_name, 'val')
dir_save_test = os.path.join(dir_name, 'test')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)
if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)
if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

## in order to pick frame data randomly, using this method.
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

##
# seek 으로 이미지 프레임(인덱스 개념)을 이동시킨다.
# 인덱스 데이터는, 이전에 id_frame 으로 랜덤하게 설정한 인덱스에 의존해서 결정된다.
offset_nframe=0

# train data
for i in range(nframe_train):
    # set current img frame position
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    # change img to array data
    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    # save label and input data(binary type)
    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

##
# validation data
offset_nframe = nframe_train

for i in range(nframe_val):
    # set current img frame position
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    # change img to array data
    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    # save label and input data(binary type)
    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

##
# Test data

offset_nframe = nframe_test + nframe_val

for i in range(nframe_test):
    # set current img frame position
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    # change img to array data
    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    # save label and input data(binary type)
    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)

## show and check the data set.

# idx = 10
# plt.figure(figsize=(10,5))
#
# img1 = np.load(os.path.join(dir_save_train, 'label_%03d.npy' % idx ))
# plt.subplot(121)
# plt.imshow(img1, cmap='gray')
# plt.title('label')
#
# img2 = np.load(os.path.join(dir_save_train, 'input_%03d.npy' % idx ))
# plt.subplot(122)
# plt.imshow(img2, cmap='gray')
# plt.title('input')
#
# plt.show()

