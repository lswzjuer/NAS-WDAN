import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from .paths import Path
from torchvision import transforms
from .transform import custom_transforms as tr
import random

def sp(args, split='train'):
    root=Path.db_root_dir('cityscapes')
    split="train"
    images_base = os.path.join(root, 'leftImg8bit', split)
    rootdir=images_base
    suffix='.png'
    
    ls =   [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]
    random.shuffle(ls)
    split = 2975//2

    return CityscapesSegmentation(args, split='train', part=ls[split:]), CityscapesSegmentation(args, split='train', part=ls[:split])

class CityscapesSegmentation(data.Dataset):
    NUM_CLASSES = 19
    def __init__(self, args, root=Path.db_root_dir('cityscapes'), split="train", part=None):
        self.NUM_CLASSES = 19
        self.root = root
        self.split = split
        self.args = args
        self.files = {}
        self.part=part
        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine', self.split)

        if self.split=="train":
            self.files[split] = part
        else:
            self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')
        

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')

        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)
        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'label': _target}

        if self.split == 'train':
            return self.transform_tr(_img,_target)
        else:
            return self.transform_val(_img,_target)


    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_tr(self, img,mask):
        composed_transforms = tr.Compose([
            FixedResize(resize=self.args.resize),
            RandomCrop(crop_size=self.args.crop_size),
            #tr.RandomGaussianBlur(),
            tr.Totensor_Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

        return composed_transforms(img,mask)

    def transform_val(self, img,mask):

        composed_transforms = tr.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Totensor_Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        return composed_transforms(img,mask)

    def transform_ts(self, img,mask):

        composed_transforms = tr.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Totensor_Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        return composed_transforms(img,mask)
    
    
# resize to 512*1024    
class FixedResize(object):
    """change the short edge length to size"""
    def __init__(self, resize=512):
        self.size1 = resize  # size= 512 

    def __call__(self, img,mask):
        assert img.size == mask.size
        w, h = img.size
        if w > h:
            oh = self.size1
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.size1
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow,oh), Image.BILINEAR)
        mask = mask.resize((ow,oh), Image.NEAREST)
        return img,mask
    
# random corp 321*321 
class RandomCrop(object):
    def __init__(self,  crop_size=321):
        self.crop_size = crop_size

    def __call__(self, img,mask):
        assert img.size==mask.size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return img,mask

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

