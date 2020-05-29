from .cityscapes import CityscapesSegmentation
#from .coco import COCOSegmentation
from .pascal_voc import VOCSegmentation
from .sbd import SBDSegmentation
from .isic2018 import ISIC2018Segmentation
from .paths import Path
from .cvc import CVCSegmentation
from torch.utils.data import DataLoader
from .refuge import EYESegmentation
from .chaos import CHAOSegmentation
from .total_images import TotalSegmentation
from .lits19 import LITS19Segmentation

datasets_dict={
    'cityscapes':CityscapesSegmentation,
    #"coco":COCOSegmentation,
    'voc':VOCSegmentation,
    'sbd':SBDSegmentation,
    'isic2018':ISIC2018Segmentation,
    'cvc':CVCSegmentation,
    'refuge':EYESegmentation,
    'chaos':CHAOSegmentation,
    "total":TotalSegmentation,
    "lits19":LITS19Segmentation
}


def get_dataloder(args,year="2017",split_flag="train"):
    '''
    :return: the dataloader of special datasets
    '''
    datasets_name=args.dataset.lower()
    assert datasets_name in datasets_dict.keys(),"The dataset use {} is not exist ".format(datasets_name)
    root=Path.db_root_dir(datasets_name)
    if datasets_name != "coco":
        if datasets_name=="total":
            dataset = datasets_dict[datasets_name](args=args, split=split_flag)
        else:
            dataset=datasets_dict[datasets_name](args=args,base_dir=root,split=split_flag)
    else:
        dataset=datasets_dict[datasets_name](args=args,base_dir=root,split=split_flag,year=year)

    if split_flag=="train":
        batch_size=args.train_batch
        shuffle=True
        num_workers = args.num_workers
    else:
        batch_size=args.val_batch
        shuffle=False
        num_workers = args.num_workers

    dataloder=DataLoader(dataset=dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers,
                         pin_memory=True,
                         )
    return dataloder



if __name__=="__main__":
    import argparse
    from PIL import  Image
    import numpy as np
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 256
    args.crop_size = 256
    args.dataset='cvc'
    args.train_batch=1
    args.val_batch=1
    args.num_workers=0
    dataloader=get_dataloder(args,split_flag="train")
    count=0
    for i,sample in enumerate(dataloader):
        images,labels=sample
        image=images.numpy()[0].transpose(1,2,0).astype(np.uint8)
        label=labels.numpy()[0][0]
        image=Image.fromarray(image)
        image.show("Image")
        label[label==1]=255
        label=Image.fromarray(label)
        label.show("label")
        count+=1
        if count>10:
            break
