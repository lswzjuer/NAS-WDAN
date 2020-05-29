import numpy as np
import torch
import torch.nn as nn
import os
import pydicom
from PIL import  Image
import cv2




def transform_dataset(root,new_root, dirname):
    images_path = os.path.join(root, dirname, 'DICOM_anon')
    mask_path = os.path.join(root, dirname, 'Ground')
    new_images_path=os.path.join(new_root,dirname,'DICOM_anon')
    new_masks_path=os.path.join(new_root,dirname,'Ground')
    if not os.path.exists(new_images_path):
        os.makedirs(new_images_path)
    if not os.path.exists(new_masks_path):
        os.makedirs(new_masks_path)

    # old images` path
    images = os.listdir(images_path)
    for image_name in images:
        if 'IMG' in image_name:
            image_index=int(image_name[:-4].split('-')[-1][2:])-1
            image_index=str(image_index).rjust(3,'0')
            image_mask_name = 'liver_GT_' + str(image_index) + '.png'
        else:
            image_mask_name = 'liver_GT_' + image_name[:-4].split(',')[0][2:] + '.png'
        img_path = os.path.join(images_path, image_name)
        img_mask_path = os.path.join(mask_path, image_mask_name)
        # new path
        new_img_path=os.path.join(new_images_path,image_mask_name)
        new_mask_path=os.path.join(new_masks_path,image_mask_name)
        img = pydicom.dcmread(img_path,force=True)
        # import matplotlib.pyplot as plt
        # plt.imshow(img.pixel_array, "gray")
        # plt.show()
        # v=m*array+b
        img, itercept = img.RescaleSlope * img.pixel_array + img.RescaleIntercept, img.RescaleIntercept
        img[img >= 4000] = itercept  # C

        # 512*512
        mask = Image.open(img_mask_path).convert('L')
        # 512*512
        mask.save(new_mask_path)
        img=Image.fromarray(img).convert('L')
        img.save(new_img_path)




def window(arr,wl=-600,ww=1500):
    '''
    :param arr:
    :param wl:
    :param ww:
    :return:
    '''
    lb=wl-ww/2
    arr=(arr-lb)/ww
    arr[arr>1]=1
    arr[arr<0]=0
    return arr


def transform_dataset_v2(root,new_root, dirname):
    images_path = os.path.join(root, dirname, 'DICOM_anon')
    mask_path = os.path.join(root, dirname, 'Ground')
    new_images_path=os.path.join(new_root,dirname,'DICOM_anon')
    new_masks_path=os.path.join(new_root,dirname,'Ground')
    if not os.path.exists(new_images_path):
        os.makedirs(new_images_path)
    if not os.path.exists(new_masks_path):
        os.makedirs(new_masks_path)

    # old images` path
    images = os.listdir(images_path)
    for image_name in images:
        if 'IMG' in image_name:
            image_index=int(image_name[:-4].split('-')[-1][2:])-1
            image_index=str(image_index).rjust(3,'0')
            image_mask_name = 'liver_GT_' + str(image_index) + '.png'
        else:
            image_mask_name = 'liver_GT_' + image_name[:-4].split(',')[0][2:] + '.png'
        img_path = os.path.join(images_path, image_name)
        img_mask_path = os.path.join(mask_path, image_mask_name)
        # new path
        new_img_path=os.path.join(new_images_path,image_mask_name)
        new_mask_path=os.path.join(new_masks_path,image_mask_name)
        img = pydicom.dcmread(img_path,force=True)
        # import matplotlib.pyplot as plt
        # plt.imshow(img.pixel_array, "gray")
        # plt.show()
        # v=m*array+b
        img, itercept = img.RescaleSlope * img.pixel_array + img.RescaleIntercept, img.RescaleIntercept
        #img[img >= 4000] = itercept  # C
        img=window(img)
        img=img*255

        # 512*512
        mask = Image.open(img_mask_path).convert('L')
        # 512*512
        mask.save(new_mask_path)
        img=Image.fromarray(img).convert('L')
        img.save(new_img_path)





def main():


    dataset_dir=r'C:\Users\rileyliu\Desktop\CHAO'
    old_root='train'
    new_root='transform_train_v1'

    subdirs=os.listdir(os.path.join(dataset_dir,old_root))
    for subdir in subdirs:
        transform_dataset_v2(os.path.join(dataset_dir,old_root),os.path.join(dataset_dir,new_root),subdir)



if __name__ == '__main__':
    main()
