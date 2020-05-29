from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
import random
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
from .paths import Path
from .transform import custom_transforms as tr
import torchvision.transforms.functional as tf
import tifffile as tiff
import torch
import random


#DATALIST = ['isic2018', 'cvc', 'chaos']
#DATALIST = ['isic2018', 'cvc']
#DATALIST = ['isic2018']

#DATALIST = ['isic2018', 'cvc']
#DATALIST = ['isic2018', 'chaos']
DATALIST = ['cvc', 'chaos']

class TotalSegmentation(Dataset):
    """
    ISIC2018 dataset
    """
    def __init__(self,
                 args,
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply     CXHXW   1XHXW
        """
        super().__init__()
        self.flag=split
        self.args = args
        self.images=[]
        self.masks=[]
        self.ids=[]
        for dataset in DATALIST:
            if dataset=="isic2018":
                base_dir=Path.db_root_dir(dataset)
                image_dir = os.path.join(base_dir, self.flag)
                cat_dir = os.path.join(base_dir, self.flag + "_GT")
                filenames = os.listdir(image_dir)
                for filename in filenames:
                    ext = os.path.splitext(filename)[-1]
                    index_name=filename.split(".")[0]
                    filename = filename.split('_')[-1][:-len('.jpg')]
                    if ext == '.jpg':
                        self.ids.append(index_name)
                        self.images.append(os.path.join(image_dir,'ISIC_' + filename + '.jpg'))
                        self.masks.append(os.path.join(cat_dir,'ISIC_' + filename + '_segmentation.png'))
            if dataset=="cvc":
                base_dir=Path.db_root_dir(dataset)
                image_dir = os.path.join(base_dir, self.flag)
                cat_dir = os.path.join(base_dir, self.flag + "_GT")
                filenames = os.listdir(image_dir)
                filenames = sorted(filenames, key=lambda x: int(x.split('.')[0]))
                for filename in filenames:
                    index_name=filename.split(".")[0]
                    self.images.append(os.path.join(image_dir,filename))
                    self.masks.append(os.path.join(cat_dir,filename))
                    self.ids.append(index_name)
                    assert os.path.splitext(filename)[-1] == '.tif'
            if dataset=="chaos":
                base_dir=Path.db_root_dir(dataset)
                image_dir=os.path.join(base_dir,"transform_" + self.flag+"_v1")
                subdirs = os.listdir(image_dir)
                for subdir in subdirs:
                    sub_images_path = os.path.join(image_dir, subdir, 'DICOM_anon')
                    sub_masks_path = os.path.join(image_dir, subdir, 'Ground')
                    sub_images = os.listdir(sub_images_path)
                    for image_name in sub_images:
                        image_index = image_name.split('.')[0]
                        img_path = os.path.join(sub_images_path, image_name)
                        img_mask_path = os.path.join(sub_masks_path, image_name)
                        self.images.append(img_path)
                        self.masks.append(img_mask_path)
                        self.ids.append(image_index)
        # Display stats
        # print(self.images)
        # print(self.images)
        # print(self.ids)
        # shuffle
        random.seed(10)
        index_list=[i for i in range(len(self.images))]
        random.shuffle(index_list)
        self.images=[self.images[index] for index in index_list]
        self.masks=[self.masks[index] for index in index_list]
        self.ids=[self.ids[index] for index in index_list]

        print('Number of images in {}: {:d}'.format(split, len(self.images)))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name,mask_name,index_name=self.images[index],self.masks[index],self.ids[index]
        ext=image_name.split('.')[-1]
        if ext=="jpg":
            # isic2018 dataset
            _img = Image.open(image_name).convert('RGB')
            _target = Image.open(mask_name)
            # 0,255 ---> 0,1
            _target = (np.asarray(_target) > 0).astype(np.float32)
            _target = Image.fromarray(_target)
            if self.flag=="train":
                composed_transforms = tr.Compose([
                    # tr.RandomVerticallyFlip(),
                    ##tr.RandomHorizontalFlip(),
                    # tr.RandomRotate(),
                    tr.RandomScaleCrop(base_size=256, crop_size=256),
                    # tr.RandomColorJitter(),
                    # tr.RandomGaussianBlur(),
                    tr.Totensor_Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ])
                image,mask=composed_transforms(_img,_target)
                return image, mask, index_name

            else:
                composed_transforms = tr.Compose([
                    tr.FixScaleCrop(crop_size=256),
                    tr.Totensor_Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ])
                # c,h,w     1,h,w
                image,mask=composed_transforms(_img,_target)
                return image, mask, index_name

        elif ext=="tif":
            # cvc dataset
            # rgb single channel
            # _img=cv2.imread(self.data_list[index],cv2.IMREAD_COLOR)
            # _img=_img[:,:,[2,1,0]]
            # _img=Image.fromarray(_img).convert("RGB")
            # true rgb
            _img = tiff.imread(image_name)
            _img = Image.fromarray(_img).convert("RGB")
            # 0....255
            _target = Image.open(mask_name)
            _target=(np.asarray(_target)/255).astype(np.float32)
            _target=Image.fromarray(_target)
            if self.flag=="train":
                composed_transforms = tr.Compose([
                    # tr.RandomVerticallyFlip(),
                    ##tr.RandomHorizontalFlip(),
                    ##tr.RandomRotate(),
                    # tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
                    # tr.FixScaleCrop(crop_size=self.args.crop_size),
                    tr.FixedResize(size=(256, 256)),
                    # tr.RandomColorJitter(),
                    # tr.RandomGaussianBlur(),
                    # mask:0...255 ---> 1,h,w 0...1
                    tr.Totensor_Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ])
                image,mask=composed_transforms(_img,_target)
                return image, mask, index_name

            else:
                composed_transforms = tr.Compose([
                    tr.FixedResize(size=(256, 256)),
                    tr.Totensor_Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ])
                image,mask=composed_transforms(_img, _target)
                return image, mask, index_name

        elif ext=="png":
            # h,w
            _image = Image.open(image_name).convert('L')
            _mask = Image.open(mask_name).convert('L')
            # _mask = (np.asarray(_mask) > 0).astype(np.float32)
            # _mask = Image.fromarray(_mask)

            if self.flag=="train":
                composed_transforms = tr.Compose([
                    # tr.RandomColorJitter(),
                    # tr.RandomGaussianBlur(),
                    # tr.RandomScaleCrop(base_size=300, crop_size=256),
                    tr.FixedResize(size=(256, 256)),
                    #tr.RandomVerticallyFlip(),
                    #tr.RandomHorizontalFlip(),
                    # tr.FixScaleCrop(crop_size=self.args.crop_size),
                    # tr.FixedResize(size=(192,256)),
                    # tr.RandomColorJitter(),
                    # tr.RandomGaussianBlur(),
                    # tr.Totensor_Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                    tr.RandomElasticTransform(alpha=1.5, sigma=0.07, img_type="L")
                ])
                # 256x256  256x256 Image array
                image, mask = composed_transforms(_image, _mask)
                # h,w-->1 h,w--> /255-->mean,std normalize   --> n,3,h,w
                image = tf.to_tensor(image)
                ##image = tf.normalize(image, mean=[0.2389], std=[0.2801])
                image=image.repeat(3,1,1)
                # h,w  --> 1,h,w --> bool                  -----> n,h,w
                mask = torch.from_numpy(np.asarray(mask).astype(np.float32)).unsqueeze(0)
                mask[mask == 255] = 1
                return image, mask, index_name
            else:
                composed_transforms = tr.Compose([
                    tr.FixedResize(size=(256,256)),
                ])
                # 256x256  256x256 Image array
                image, mask = composed_transforms(_image, _mask)
                # h,w-->1 h,w--> /255-->mean,std normalize   --> n,1,h,w
                image = tf.to_tensor(image)
                ##image = tf.normalize(image,  mean=[0.2389], std=[0.2801])
                image=image.repeat(3,1,1)
                # expand to dims 3
                # h,w  --> h,w --> bool                  -----> n,h,w
                mask = torch.from_numpy(np.asarray(mask).astype(np.float32)).unsqueeze(0)
                mask[mask == 255] = 1
                return image,mask,index_name

    def __str__(self):
        return 'TOTAL(split=' + str(self.flag) + ')'




def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)


    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    shuffle_x= x[index]
    return mixed_x, shuffle_x, y_a, y_b, lam





if __name__=="__main__":
    import argparse
    from torch.utils.data import DataLoader
    import torch
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    dataset=TotalSegmentation(args,split="train")
    dataloader=DataLoader(dataset,batch_size=20,num_workers=2,shuffle=True)
    count=0
    for i,sample in enumerate(dataloader):
        images,labels,_=sample
        mixed_images,shuffle_x,labels,perm_labels,lam=mixup_data(images,labels,use_cuda=False)
        #
        # original_image=images.numpy()[0].transpose(1,2,0)*255
        # original_image=original_image.astype(np.uint8)
        # original_image=Image.fromarray(original_image)
        # original_image.show("original_Image")
        #
        # shuffle_image=shuffle_x.numpy()[0].transpose(1,2,0)*255
        # shuffle_image=shuffle_image.astype(np.uint8)
        # shuffle_image=Image.fromarray(shuffle_image)
        # shuffle_image.show("shuffle_Image")


        image=mixed_images.numpy()[0].transpose(1,2,0)*255
        image=image.astype(np.uint8)
        # n,1,h,w
        label=labels.numpy()[0][0].astype(np.uint8)
        image=Image.fromarray(image)
        image.show("mixed_Image")
        # label[label==1]=255
        # label=Image.fromarray(label)
        # label.show("label1")
        #
        # plabel=perm_labels.numpy()[0][0].astype(np.uint8)
        # plabel[plabel==1]=255
        # plabel=Image.fromarray(plabel)
        # plabel.show("label2")

        count+=1
        if count>10:
            break

