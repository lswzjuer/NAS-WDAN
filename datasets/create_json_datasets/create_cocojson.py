import numpy as np
import os
import cv2
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
from pycococreatortools import pycococreatortools
import json




INFO = {
    "description": "Kaggle Dataset",
    "url": "https://github.com/",
    "version": "0.1.0",
    "year": 2020,
    "contributor": "ycl",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'cell',
        'supercategory': 'cell',
    },
]


def rleencode(mask):
    '''
    :param mask: H*W
    :return:
    '''
    dots = np.where(mask.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    run_lengths = ' '.join([str(r) for r in run_lengths])
    return run_lengths


def rledecode(mask_rle, shape=(768, 768)):
    s = mask_rle.split()
    starts = np.asarray(s[0::2], dtype=int)
    lengths = np.asarray(s[1::2], dtype=int)
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction



def write2csv(file, ImageId, EncodedPixels):
    df = pd.DataFrame({'ImageId': ImageId, 'EncodedPixels': EncodedPixels})
    df.to_csv(file, index=False, columns=['ImageId', 'EncodedPixels'])



# total_image_path, rle_code
def create_mask_csv(images_path,csv_path):
    '''
    :param images_path:
    :return:
    '''
    def get_mask_path(im_path):
        image_name=os.path.basename(im_path)
        sub_dir_path=os.path.dirname(os.path.dirname(im_path))
        mask_dir_name=image_name.split('.')[0]+"_outlines"
        mask_dir_path=os.path.join(sub_dir_path,"crops",mask_dir_name)
        mask_image_name=os.listdir(mask_dir_path)
        mask_image_name=[ i for i in mask_image_name if len(i.split('_'))==3 ]
        mask_image_path=[ os.path.join(mask_dir_path,i) for i in mask_image_name]
        return mask_image_path

    print("Total image len:{}".format(images_path))
    ImageId_d = []
    EncodedPixels_d = []
    for i in range(len(images_path)):
        # './new_training/76-9-10052-crop2/xy-images/crop2-Composite-10051.tif'
        # in train2020_new     76-9-10052-crop2-Composite-10051.png
        print("Process: {}/{}".format(i,len(images_path)))
        im_path=images_path[i]
        print(im_path)
        masks_path=get_mask_path(im_path)
        for m_path in tqdm(masks_path):
            mask = plt.imread(m_path)
            mask_rle=rleencode(mask)
            ImageId_d.append(im_path)
            EncodedPixels_d.append(mask_rle)
    write2csv(csv_path, ImageId_d, EncodedPixels_d)




def save_bad_ann(save_dir,imagepath, mask, segmentation_id):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    new_mask=mask.copy()
    new_mask[new_mask>0]=255
    img = Image.open(imagepath)
    img=np.asarray(img).astype(np.uint8)
    fig, axarr = plt.subplots(1, 3)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[2].axis('off')
    axarr[0].imshow(img)
    axarr[1].imshow(new_mask)
    axarr[2].imshow(img)
    axarr[2].imshow(new_mask, alpha=0.4)
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    image_name=os.path.basename(imagepath).split('.')[0]+'_'+str(segmentation_id)+'.png'
    save_path=os.path.join(save_dir,image_name)
    plt.savefig(save_path)
    plt.close()


def create_json_direct(images_path,new_path):
    '''
    :param images_path:
    :param new_path:
    :return:
    '''

    def get_mask_path(im_path):
        image_name=os.path.basename(im_path)
        sub_dir_path=os.path.dirname(os.path.dirname(im_path))
        mask_dir_name=image_name.split('.')[0]+"_outlines"
        mask_dir_path=os.path.join(sub_dir_path,"crops",mask_dir_name)
        mask_image_name=os.listdir(mask_dir_path)
        mask_image_name=[ i for i in mask_image_name if len(i.split('_'))==3 ]
        mask_image_path=[ os.path.join(mask_dir_path,i) for i in mask_image_name]
        return mask_image_path

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    train_images_path=os.path.join(new_path,"train2020_scale")
    if not os.path.exists(train_images_path):
        os.mkdir(train_images_path)

    train_anno_path=os.path.join(new_path,"annotations","instances_cell_train2020_scale.json")
    train_image_bad_path=os.path.join(new_path,"bad_ann")


    images_num=len(images_path)
    image_id = 1
    segmentation_id = 1
    print("Total image len:{}".format(images_num))

    for im_path in images_path:
        # new_image_name is the new name of this image in train2020_sacle and coco json file
        # './new_training/76-9-10052-crop2/xy-images/crop2-Composite-10051.tif'
        print("=====================================")

        print(im_path)
        dir_name=im_path.split('/')[-3]
        im=os.path.basename(im_path).split('.')[0]
        im='-'.join(im.split('-')[-2:])
        new_image_name=dir_name+"-"+im+".png"
        print(new_image_name)

        # read image and save to new dir
        # image.size 是 w,h
        image=Image.open(im_path)
        image_info = pycococreatortools.create_image_info(
            image_id, new_image_name, image.size)
        coco_output["images"].append(image_info)
        print("image_info:\n{}".format(image_info))

        # save to new dir
        new_image_path=os.path.join(train_images_path,new_image_name)
        image.save(new_image_path)
        print("Save as:{}".format(new_image_path))

        # find this image`s masks and make boxes & segmentation info
        masks_path=get_mask_path(im_path)
        print("Masks num of image:{}".format(len(masks_path)))

        for c in range(len(masks_path)):
            ms_path=masks_path[c]
            # ms_path is one of the masks path for  this image
            binary_mask=np.asarray(Image.open(ms_path).convert("L")).astype(np.uint8)
            binary_mask[binary_mask>0]=1
            class_id=1
            category_info = {'id': class_id, 'is_crowd': 0}
            annotation_info = pycococreatortools.create_annotation_info(
                segmentation_id, image_id, category_info, binary_mask,
                image.size, tolerance=2)

            print("image size:{} {}  mask size:{}".format(image.size[1],image.size[0],binary_mask.shape))
            print("Annotation info:{}".format(annotation_info))

            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)
            else:
                # coco_output["annotations"].append(annotation_info)
                print("There is an bad binary mask in :{}-{}".format(new_image_name,c))
                save_bad_ann(train_image_bad_path,new_image_path, binary_mask, segmentation_id)
            segmentation_id+=1
        image_id+=1

    with open(train_anno_path, 'w') as output_json_file:
        # json.dump(coco_output, output_json_file)
        json.dump(coco_output, output_json_file, indent=4)


if __name__=="__main__":

    origin_dir='/extracephonline/medai_data2/rileyliu/datasets/RM007_76/new_training'
    csv_path='./train2020-new.csv'
    new_path='/extracephonline/medai_data2/rileyliu/maskscoringrcnn/datasets/cell'


    # the final train image name pri
    images_path=[]
    sub_dir=os.listdir(origin_dir)
    for dir in sub_dir:
        im_path=os.path.join(origin_dir,dir,"xy-images")
        images=os.listdir(im_path)
        print("dir:{} image len:{}".format(dir,len(images)))
        images=[os.path.join(im_path,i) for i in images]
        images_path+=images
    # print(images_path)
    # all images` path
    # create_mask_csv(images_path)
    create_json_direct(images_path,new_path)







