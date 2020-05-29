import numpy as np
import torch
from torch.utils.data import Dataset
from .paths import Path
from tqdm import trange
import os
from pycocotools.coco import COCO
from pycocotools import mask
from torchvision import transforms
from .transform import custom_transforms as tr
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class COCOSegmentation(Dataset):
    NUM_CLASSES = 21
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
        1, 64, 20, 63, 7, 72]

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('coco'),
                 split='train',
                 year='2017'):
        super().__init__()
        ann_file = os.path.join(base_dir, 'annotations/instances_{}{}.json'.format(split, year))
        ids_file = os.path.join(base_dir, 'annotations/{}_ids_{}.pth'.format(split, year))
        self.img_dir = os.path.join(base_dir, 'images/{}{}'.format(split, year))
        self.split = split
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)
        self.args = args

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        if self.split == "train":
            return self.transform_tr(_img,_target)
        else:
            return self.transform_val(_img,_target)

    def _make_img_gt_point_pair(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        # original image RGB
        _img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        # the image`s mask which pixel value is label num
        _target = Image.fromarray(self._gen_seg_mask(
            cocotarget, img_metadata['height'], img_metadata['width']))
        return _img, _target

    # 过滤出符合像素要求的图片的ids
    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while. " + \
              "But don't worry, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
                                      img_metadata['width'])
            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        torch.save(new_ids, ids_file)
        return new_ids

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    # train aug
    def transform_tr(self, img,mask):
        composed_transforms = tr.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.data.base_size, crop_size=self.args.data.crop_size),
            tr.RandomGaussianBlur(),
            tr.Totensor_Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        return composed_transforms(img,mask)

    # val aug
    def transform_val(self, img,mask):
        composed_transforms = tr.Compose([
            tr.FixScaleCrop(crop_size=self.args.data.crop_size),
            tr.Totensor_Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        return composed_transforms(img,mask)

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