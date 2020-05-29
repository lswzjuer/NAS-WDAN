import os
import random
import numpy as np
random.seed(1)

import matplotlib.pyplot as plt
import my_function as f
import pandas as pd
from PIL import Image
import cv2 as cv

imgs_dir = '/data/home/luyanliu/external-data'
img_names = os.listdir(imgs_dir)
random.shuffle(img_names)
print(img_names[0])
number_case = img_names.__len__()
number_train = number_case
#number_train = np.int32(np.ceil(number_case/10 * 9))
print(number_case)
print(number_train)



## train sets
ImageId_d = []
EncodedPixels_d = []
for i in range(number_train):
    print(i)
    mask_names = os.listdir(os.path.join(imgs_dir,img_names[i],'masks'))
    if img_names[i] == '170bc41b2095177cccd3d4c8977c619147580f1d93b4fe9701eddd77736d4ece':
        img_path = os.path.join(imgs_dir,img_names[i],'images',img_names[i]+'.jpeg')
    else:
        img_path = os.path.join(imgs_dir, img_names[i], 'images', img_names[i]+'.png')
    #img_path = os.path.join(imgs_dir, img_names[i], 'images', img_names[i]+'.tif')
    img = cv.imread(img_path)
    cv.imwrite('/data/home/luyanliu/external2020/' + img_names[i] + '.png',img)
    for j in range(len(mask_names)):
        mask_path = os.path.join(imgs_dir,img_names[i],'masks',mask_names[j])
        mask = plt.imread(mask_path)
        ImageId_batch, EncodedPixels_batch, _ = f.numpy2encoding(mask, img_names[i], scores=None,
                                                                 dilation=False)
        ImageId_d += ImageId_batch
        EncodedPixels_d += EncodedPixels_batch

f.write2csv('train-external.csv', ImageId_d, EncodedPixels_d)


# test sets
#ImageId_d = []
#EncodedPixels_d = []
#or i in range(number_train,number_case):
#   print(i)
#   mask_names = os.listdir(os.path.join(imgs_dir,img_names[i],'masks'))
#   if img_names[i] == '170bc41b2095177cccd3d4c8977c619147580f1d93b4fe9701eddd77736d4ece':
#       img_path = os.path.join(imgs_dir,img_names[i],'images',img_names[i]+'.jpeg')
#   else:
#       img_path = os.path.join(imgs_dir, img_names[i], 'images', img_names[i]+'.png')
#   img = cv.imread(img_path)
#   cv.imwrite('/DSB2018/data/val2019/' + img_names[i] + '.png',img)
#   for j in range(len(mask_names)):
#       mask_path = os.path.join(imgs_dir,img_names[i],'masks',mask_names[j])
#       mask = plt.imread(mask_path)
#       ImageId_batch, EncodedPixels_batch, _ = f.numpy2encoding(mask, img_names[i], scores=None,
#                                                                dilation=False)
#       ImageId_d += ImageId_batch
#       EncodedPixels_d += EncodedPixels_batch
#.write2csv('submission_val.csv', ImageId_d, EncodedPixels_d)


