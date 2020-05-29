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


def transform_baseline(args,image,GT,augmentation_prob=0.5,mode="train"):
    assert image.size==GT.size
    RotationDegrees = [0, 90, 180, 270]
    aspect_ratio = image.size[1] / image.size[0]
    Transform = []
    ResizeRange = random.randint(300, 320)
    Transform.append(T.Resize((int(ResizeRange * aspect_ratio), ResizeRange)))
    p_transform = random.random()
    if (mode == 'train') and p_transform <= augmentation_prob:
        RotationDegree = random.randint(0, 3)
        RotationDegree = RotationDegrees[RotationDegree]
        if (RotationDegree == 90) or (RotationDegree == 270):
            aspect_ratio = 1 / aspect_ratio
        Transform.append(T.RandomRotation((RotationDegree, RotationDegree)))
        RotationRange = random.randint(-10, 10)
        Transform.append(T.RandomRotation((RotationRange, RotationRange)))
        CropRange = random.randint(250, 270)
        Transform.append(T.CenterCrop((int(CropRange * aspect_ratio), CropRange)))
        Transform = T.Compose(Transform)
        image = Transform(image)
        GT = Transform(GT)
        ShiftRange_left = random.randint(0, 20)
        ShiftRange_upper = random.randint(0, 20)
        ShiftRange_right = image.size[0] - random.randint(0, 20)
        ShiftRange_lower = image.size[1] - random.randint(0, 20)
        image = image.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))
        GT = GT.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))
        if random.random() < 0.5:
            image = F.hflip(image)
            GT = F.hflip(GT)
        if random.random() < 0.5:
            image = F.vflip(image)
            GT = F.vflip(GT)
        Transform = T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.02)
        image = Transform(image)
        Transform = []
    Transform.append(T.Resize((args.crop_size, args.crop_size)))
    Transform.append(T.ToTensor())
    Transform = T.Compose(Transform)
    image = Transform(image)
    GT = Transform(GT)
    Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    image = Norm_(image)
    return image, GT


class ISIC2018Segmentation(Dataset):
    """
    ISIC2018 dataset
    """
    NUM_CLASSES = 2
    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('isic2018'),
                 split='train',
                 is_baseline_transform=False,
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply     CXHXW   1XHXW
        """
        super().__init__()
        self.is_baseline_transform=is_baseline_transform
        self.flag=split
        self.args = args
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, self.flag)
        self._cat_dir = os.path.join(self._base_dir, self.flag+"_GT")

        self.filenames = os.listdir(self._image_dir)
        self.data_list = []
        self.gt_list = []
        self.img_ids=[]
        self.filenames=sorted(self.filenames,key=lambda x: int(x.split('_')[-1][:-len('.jpg')]))
        for filename in self.filenames:
            ext = os.path.splitext(filename)[-1]
            if ext == '.jpg':
                filename = filename.split('_')[-1][:-len('.jpg')]
                self.img_ids.append(filename)
                self.data_list.append('ISIC_' + filename + '.jpg')
                self.gt_list.append('ISIC_' + filename + '_segmentation.png')

        assert (len(self.data_list) == len(self.gt_list))
        self.data_list=[os.path.join(self._image_dir,i) for i in self.data_list]
        self.gt_list=[os.path.join(self._cat_dir,i) for i in self.gt_list]
        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.data_list)))


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        # change the _target value range, the transform just change the value type
        if not self.is_baseline_transform:
            if self.flag == "train":
                image, mask = self.transform_tr(_img, _target)
                return image, mask, self.img_ids[index]
            else:
                image, mask = self.transform_val(_img, _target)
                #image, mask = self.transform_test(_img, _target)
                return image, mask, self.img_ids[index]
        else:
            if self.flag == "train":
                image, mask = self.transform_tr_baseline(_img, _target,mode="train")
                return image, mask, self.img_ids[index]
            else:
                image, mask = self.transform_tr_baseline(_img, _target,mode="valid")
                return image, mask, self.img_ids[index]


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.data_list[index]).convert('RGB')
        _target = Image.open(self.gt_list[index])
        _target=(np.asarray(_target)>0).astype(np.float32)
        _target=Image.fromarray(_target)
        return _img, _target


    def transform_tr(self,img,mask):
        composed_transforms=tr.Compose([
            # tr.RandomVerticallyFlip(),
            tr.RandomHorizontalFlip(),
            # tr.RandomRotate(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            #tr.RandomColorJitter(),
            #tr.RandomGaussianBlur(),
            tr.Totensor_Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        return composed_transforms(img,mask)

    def transform_val(self, img,mask):
        composed_transforms = tr.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Totensor_Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        return composed_transforms(img,mask)

    def transform_test(self, img,mask):
        composed_transforms = tr.Compose([
            tr.FixedResize(size=(256, 256)),
            tr.Totensor_Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        return composed_transforms(img,mask)

    def transform_tr_v2(self,img,mask):
        composed_transforms = tr.Compose([
            tr.FixedResize(size=(256,256)),  # h,w
            tr.Totensor_Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        return composed_transforms(img,mask)

    def transform_tr_baseline(self,img,mask,mode):
        return transform_baseline(self.args,img,mask,mode=mode)

    def __str__(self):
        return 'ISIC2018(split=' + str(self.flag) + ')'


if __name__=="__main__":
    import argparse
    from torch.utils.data import DataLoader
    import torch
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 256
    args.crop_size = 256
    args.dataset='isic2018'
    dataset=ISIC2018Segmentation(args,split="train",is_baseline_transform=False)
    dataloader=DataLoader(dataset,batch_size=2,num_workers=2,shuffle=True)
    count=0
    for i,sample in enumerate(dataloader):
        images,labels,_=sample
        print(torch.max(labels))
        labels = labels == torch.max(labels)
        image=images.numpy()[0].transpose(1,2,0)*255
        image=image.astype(np.uint8)
        label=labels.numpy()[0][0].astype(np.uint8)
        image=Image.fromarray(image)
        image.show("Image")
        label[label==1]=255
        label=Image.fromarray(label)
        label.show("label")
        count+=1
        if count>5:
            break
