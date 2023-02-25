##
import os
import numpy as np
import matplotlib.pyplot as plt

##
result_dir = './result/numpy'

lst_data = os.listdir(result_dir)

lst_label = [foo for foo in lst_data if foo.startswith('label')]
lst_input = [foo for foo in lst_data if foo.startswith('input')]
lst_output = [foo for foo in lst_data if foo.startswith('output')]

lst_label.sort()
lst_input.sort()
lst_output.sort()

##
id = 0

label = np.load(os.path.join(result_dir, lst_label[0]))
input = np.load(os.path.join(result_dir, lst_input[0]))
output = np.load(os.path.join(result_dir, lst_output[0]))

plt.subplot(131)
plt.imshow(label, cmap='gray')
plt.title('label')

plt.subplot(132)
plt.imshow(input, cmap='gray')
plt.title('input')

plt.subplot(133)
plt.imshow(output, cmap='gray')
plt.title('output')

plt.show()

