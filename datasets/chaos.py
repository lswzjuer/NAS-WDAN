#coding:utf-8
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf
import torch
from .paths import Path
from .transform import custom_transforms as tr
import pydicom

def make_dataset(root, dirname):
    images_path = os.path.join(root, dirname, 'DICOM_anon')
    mask_path = os.path.join(root, dirname, 'Ground')
    images = os.listdir(images_path)
    images_list = []
    for image_name in images:
        if 'IMG' in image_name:
            image_mask_name = 'liver_GT_' + image_name[:-4].split('-')[-1][2:] + '.png'
        else:
            image_mask_name = 'liver_GT_' + image_name[:-4].split(',')[0][2:] + '.png'
        img_path = os.path.join(images_path, image_name)
        img_mask_path = os.path.join(mask_path, image_mask_name)
        images_list.append((img_path, img_mask_path))
    return images_list



class CHAOSegmentation(Dataset):
    """
    CVC dataset
    """
    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('chaos'),
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply     CXHXW   1XHXW
        """
        super(CHAOSegmentation, self).__init__()
        self.flag=split
        self.args = args
        self.size=args.crop_size
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, "transform_"+self.flag+"_v1")
        self.subdirs = os.listdir(self._image_dir)
        self.mean=[0.2389]
        self.std=[0.2801]
        self.all_images=[]
        for subdir in self.subdirs:
            sub_images_path = os.path.join(self._image_dir, subdir, 'DICOM_anon')
            sub_masks_path = os.path.join(self._image_dir, subdir, 'Ground')
            sub_images = os.listdir(sub_images_path)
            for image_name in sub_images:
                image_index=image_name.split('.')[0]
                img_path = os.path.join(sub_images_path, image_name)
                img_mask_path = os.path.join(sub_masks_path, image_name)
                self.all_images.append((img_path,img_mask_path,[subdir,image_index]))
        assert len(self.all_images)!=0,"the images can`t be zero!"
        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.all_images)))


    def __len__(self):
        return len(self.all_images)


    def __getitem__(self, index):
        image_path,mask_path,image_index=self.all_images[index]
        # 512x512  512x512
        _img, _target = self._make_img_gt_point_pair(image_path,mask_path)
        if self.flag == "train":
            image,mask=self.transform_tr(_img,_target)
            return image,mask,image_index
        else:
            image,mask=self.transform_val(_img,_target)
            return image,mask,image_index

    def _make_img_gt_point_pair(self, img_path,target_path):
        '''
        :param img_path:
        :param target_path:
        :return:
        '''
        img = Image.open(img_path).convert('L')
        mask = Image.open(target_path).convert('L')
        # 512*512
        return img,mask

    def transform_tr(self,img,mask):
        composed_transforms=tr.Compose([
            # tr.RandomColorJitter(),
            # tr.RandomGaussianBlur(),
            #tr.RandomScaleCrop(base_size=300, crop_size=256),
            tr.FixedResize(size=(self.size,self.size)),
            tr.RandomVerticallyFlip(),
            tr.RandomHorizontalFlip(),
            # tr.FixScaleCrop(crop_size=self.args.crop_size),
            #tr.FixedResize(size=(192,256)),
            #tr.RandomColorJitter(),
            #tr.RandomGaussianBlur(),
            #tr.Totensor_Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            tr.RandomElasticTransform(alpha=1.5,sigma=0.07,img_type="L")
        ])
        # 256x256  256x256 Image array
        img,mask=composed_transforms(img,mask)
        # h,w-->1 h,w--> /255-->mean,std normalize   --> n,1,h,w
        img=tf.to_tensor(img)
        #img=tf.normalize(img,mean=self.mean,std=self.std)
        # h,w  --> 1,h,w --> bool                  -----> n,h,w
        mask=torch.from_numpy(np.asarray(mask).astype(np.float32))
        mask[mask==255]=1
        return img,mask


    def transform_val(self, img,mask):
        composed_transforms=tr.Compose([
            tr.FixedResize(size=(self.size,self.size)),
        ])
        # 256x256  256x256 Image array
        img,mask=composed_transforms(img,mask)
        # h,w-->1 h,w--> /255-->mean,std normalize   --> n,1,h,w
        img=tf.to_tensor(img)
        #img=tf.normalize(img,mean=self.mean,std=self.std)
        # h,w  --> h,w --> bool                  -----> n,h,w
        mask=torch.from_numpy(np.asarray(mask).astype(np.float32))
        mask[mask==255]=1
        return img,mask

    def __str__(self):
        return 'CHAOS(split=' + str(self.flag) + ')'



if __name__=="__main__":


    import argparse
    from torch.utils.data import DataLoader
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset='chaos'
    args.crop_size=256
    dataset=CHAOSegmentation(args,split="train")
    dataloader=DataLoader(dataset,batch_size=10,num_workers=2,shuffle=True)
    count=0
    for i,sample in enumerate(dataloader):
        images,labels,_=sample
        print(images.size())
        print(labels.size())
        image=images.numpy()[0][0]*255
        image=image.astype(np.uint8)
        label=labels.numpy()[0].astype(np.uint8)
        image=Image.fromarray(image)
        image.show("Image")
        label[label==1]=255
        label=Image.fromarray(label)
        label.show("label")
        count+=1
        if count>5:
            break

        # dcm = pydicom.read_file(img_path)
        # import matplotlib.pyplot as plt
        # import cv2
        # plt.imshow(dcm.pixel_array, "gray")
        # plt.show()



