from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from .paths import Path
from torchvision import transforms
from .transform import custom_transforms as tr

class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 21
    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('voc'),
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()
            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg")
                _cat = os.path.join(self._cat_dir, line + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)

        for split in self.split:
            if split == "train":
                return self.transform_tr(_img,_target)
            else:
                return self.transform_val(_img,_target)

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])
        return _img, _target

    def transform_tr(self, img,mask):
        composed_transforms = tr.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            # conver 0~255-->0~1 and -mean/std
            tr.Totensor_Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return composed_transforms(img,mask)

    def transform_val(self, img,mask):
        composed_transforms = tr.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Totensor_Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        return composed_transforms(img,mask)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'

if __name__=="__main__":
    import argparse
    from torch.utils.data import DataLoader
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 320
    args.crop_size = 320
    args.dataset='isic2018'
    dataset=ISIC2018Segmentation(args,split="train")
    dataloader=DataLoader(dataset,batch_size=1,num_workers=2,shuffle=True)
    count=0
    for i,sample in enumerate(dataloader):
        images,labels=sample
        image=images.numpy()[0].transpose(1,2,0).astype(np.uint8)
        label=labels.numpy()[0]
        image=Image.fromarray(image)
        image.show("Image")
        label[label==1]=255
        label=Image.fromarray(label)
        label.show("label")
        count+=1
        if count>10:
            break
