from __future__ import print_function, division
import os
import scipy.io
import torch.utils.data as data
from PIL import Image
from .paths import Path

from torchvision import transforms
from .transform import custom_transforms as tr

class SBDSegmentation(data.Dataset):
    NUM_CLASSES = 21

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('sbd'),
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._dataset_dir = os.path.join(self._base_dir, 'dataset')
        self._image_dir = os.path.join(self._dataset_dir, 'img')
        self._cat_dir = os.path.join(self._dataset_dir, 'cls')


        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        # Get list of all images from the split and check that the files exist
        self.im_ids = []
        self.images = []
        self.categories = []
        for splt in self.split:
            with open(os.path.join(self._dataset_dir, splt + '.txt'), "r") as f:
                lines = f.read().splitlines()

            for line in lines:
                _image = os.path.join(self._image_dir, line + ".jpg")
                _categ= os.path.join(self._cat_dir, line + ".mat")
                assert os.path.isfile(_image)
                assert os.path.isfile(_categ)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_categ)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images: {:d}'.format(len(self.images)))


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        return self.transform(_img,_target)

    def __len__(self):
        return len(self.images)

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.fromarray(scipy.io.loadmat(self.categories[index])["GTcls"][0]['Segmentation'][0])

        return _img, _target

    def transform(self, img,mask):
        composed_transforms = tr.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Totensor_Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        return composed_transforms(img,mask)

    def __str__(self):
        return 'SBDSegmentation(split=' + str(self.split) + ')'


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