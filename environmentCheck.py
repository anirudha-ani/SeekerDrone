#Checking if torch is using CUDA

import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())


#Checking if Tensorflow is using GPU

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
