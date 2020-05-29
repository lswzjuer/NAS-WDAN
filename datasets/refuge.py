from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from .paths import Path
from .transform import custom_transforms as tr
import torch
import torch.nn as nn


CROP_SIZE=[512,512]
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


class EYESegmentation(Dataset):
    """
    CVC dataset
    """
    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('refuge'),
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply     CXHXW   1XHXW
        """
        super(EYESegmentation, self).__init__()
        self.flag=split
        self.args = args
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, self.flag)
        self._cat_dir = os.path.join(self._base_dir, self.flag+"_GT")
        self.filenames = os.listdir(self._image_dir)
        # groud truth images
        self.data_list = []
        self.gt_list = []
        self.img_ids=[]
        for filename in self.filenames:
            # train image
            fileindex=filename[:5]
            self.data_list.append(filename)
            gt_name=fileindex+'.bmp'
            self.gt_list.append(gt_name)
            self.img_ids.append(fileindex)
            if split=="train":
                assert os.path.splitext(filename)[-1]=='.png'
            else:
                assert os.path.splitext(filename)[-1]=='.jpg'
        assert (len(self.data_list) == len(self.gt_list))
        self.data_list=[os.path.join(self._image_dir,i) for i in self.data_list]
        self.gt_list=[os.path.join(self._cat_dir,i) for i in self.gt_list]
        # 0 cup 1 disc 2 bk
        self.id_to_trainid= {0: 0, 128: 1, 255: 2}
        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.data_list)))


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        '''
        :param index:
        :return:
        '''
        _img, _target = self._make_img_gt_point_pair(index)
        w,h=_img.size
        _target=_target.resize((w,h), Image.NEAREST)
        assert _img.size==_target.size
        if self.flag == "train":
            images,mask=self.transform_tr(_img,_target)
            return images,mask,self.img_ids[index]
        else:
            images,mask=self.transform_val(_img,_target)
            return images,mask,self.img_ids[index]


    def _make_img_gt_point_pair(self, index):
        '''
        :param index:
        :return:
        '''
        _img=Image.open(self.data_list[index]).convert("RGB")
        _target = Image.open(self.gt_list[index]).convert("L")
        # 0-->0 cup   128-->1 disc  255-->2 bk
        return _img, _target


    def transform_tr(self,img,mask):
        '''
        :param img:
        :param mask:
        :return:
        '''
        composed_transforms=tr.Compose([
            tr.RandomVerticallyFlip(),
            tr.RandomHorizontalFlip(),
            #tr.RandomColorJitter(),
        ])
        # morphology change
        img,mask=composed_transforms(img,mask)
        image = img.resize(CROP_SIZE, Image.BICUBIC)
        label = mask.resize(CROP_SIZE, Image.NEAREST)
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)# 0 disc 128 cup 255 bk
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        # BGR image to inference
        image = image[:, :, ::-1]  # change to BGR(1280*720*3)
        image -= IMG_MEAN
        #image/=128
        image = image.transpose((2, 0, 1))
        image= torch.from_numpy(image.copy())
        label_copy=torch.from_numpy(label_copy)
        return image,label_copy


    def transform_val(self,img,mask):
        '''
        :param img:
        :param mask:
        :return:
        '''
        image = img.resize(CROP_SIZE, Image.BICUBIC)
        label = mask.resize(CROP_SIZE, Image.NEAREST)
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)# 0 disc 128 cup 255 bk
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        # BGR image to inference
        image = image[:, :, ::-1]  # change to BGR(1280*720*3)
        image -= IMG_MEAN
        # image/=128
        # print(image)
        image = image.transpose((2, 0, 1))
        image= torch.from_numpy(image.copy())
        label_copy=torch.from_numpy(label_copy)
        return image,label_copy

    def __str__(self):
        return 'REFUGE(split=' + str(self.flag) + ')'





if __name__=="__main__":

    import argparse
    from torch.utils.data import DataLoader
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset='eye'
    dataset=EYESegmentation(args,split="train")
    dataloader=DataLoader(dataset,batch_size=10,num_workers=2,shuffle=True)
    count=0
    for i,sample in enumerate(dataloader):
        images,labels,_=sample
        image=images.numpy()[0].transpose(1,2,0)+IMG_MEAN
        image=image[:,:,::-1]
        image=image.astype(np.uint8)
        label=labels.numpy()[0].astype(np.uint8)
        label[label==0]=0
        label[label==1]=128
        label[label==2]=255
        image=Image.fromarray(image)
        image.show("Image")
        label=Image.fromarray(label)
        label.show("label")
        count+=1
        if count>5:
            break
