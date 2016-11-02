import numpy as np
from scipy.misc import imread
import os

trainx = []
for f in os.listdir('/home/tim/data/small_imagenet/train_32x32'):
    if '.png' in f:
        print(f)
        trainx.append(imread('/home/tim/data/small_imagenet/train_32x32/'+f).reshape((1,32,32,3)))

trainx = np.concatenate(trainx,axis=0)

testx = []
for f in os.listdir('/home/tim/data/small_imagenet/valid_32x32'):
    if '.png' in f:
        print(f)
    testx.append(imread('/home/tim/data/small_imagenet/valid_32x32/'+f).reshape((1,32,32,3)))

testx = np.concatenate(testx,axis=0)

np.savez('/home/tim/data/small_imagenet/imgnet_32x32.npz', trainx=trainx, testx=testx)

