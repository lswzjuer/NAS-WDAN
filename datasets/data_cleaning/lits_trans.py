import numpy as np
import SimpleITK as sitk
import os
import shutil
import  cv2



# normalize
def hu_to_grayscale(volume, hu_min=None, hu_max=None):
    # Clip at max and min values if specified
    if hu_min is not None or hu_max is not None:
        volume = np.clip(volume, hu_min, hu_max)

    # Scale to values between 0 and 1
    mxval = np.max(volume)
    mnval = np.min(volume)
    im_volume = (volume - mnval) / max(mxval - mnval, 1e-3)

    # Return values scaled to 0-255 range, but *not cast to uint8*
    # Repeat three times to make compatible with color overlay
    #im_volume = 255 * im_volume
    return im_volume







total_dir=r'E:\segmentation\Lits'

batch1=os.path.join(total_dir,'Training Batch 1')
batch2=os.path.join(total_dir,'Training Batch 2')

files1=os.listdir(batch1)
files2=os.listdir(batch2)

trains=[]
masks=[]

for i in range(len(files1)):
    if "seg" in files1[i]:
        masks.append(os.path.join(batch1,files1[i]))
    else:
        trains.append(os.path.join(batch1, files1[i]))

for i in range(len(files2)):
    if "seg" in files2[i]:
        masks.append(os.path.join(batch2,files2[i]))
    else:
        trains.append(os.path.join(batch2, files2[i]))

assert len(trains)==len(masks)
print(trains)

np.random.seed(100)
index=[i for i in range(len(trains))]
np.random.shuffle(index)
train_nums=int(len(trains)*0.8)
train_indexs=index[:train_nums]
val_indexs=index[train_nums:]
train_images=[trains[i] for i in train_indexs]
train_masks=[masks[i] for i in train_indexs]

val_images=[trains[i] for i in val_indexs]
val_masks=[masks[i] for i in val_indexs]

train_path=r'F:\LITS19\train'
valid_path=r'F:\LITS19\valid'

# for i in range(len(train_images)):
#     image_name=os.path.basename(train_images[i])
#     parent_id=int(image_name.split('.')[0].split("-")[-1])
#     id_dir=os.path.join(train_path,str(parent_id))
#     if not os.path.exists(id_dir):
#         os.mkdir(id_dir)
#     images_path=os.path.join(id_dir,"images")
#     masks_path=os.path.join(id_dir,"masks")
#     if not os.path.exists(images_path):
#         os.mkdir(images_path)
#         os.mkdir(masks_path)
#     print("copy file:{}".format(parent_id))
#     print(train_images[i],os.path.join(id_dir,image_name))
#     print(train_masks[i],os.path.join(id_dir,os.path.basename(train_masks[i])))
#     shutil.copy2(train_images[i],os.path.join(id_dir,image_name))
#     shutil.copy2(train_masks[i],os.path.join(id_dir,os.path.basename(train_masks[i])))
#


# for i in range(len(val_images)):
#     image_name=os.path.basename(val_images[i])
#     print(image_name)
#     parent_id=int(image_name.split('.')[0].split("-")[-1])
#     print(parent_id)
#     id_dir=os.path.join(valid_path,str(parent_id))
#     if not os.path.exists(id_dir):
#         os.mkdir(id_dir)
#     images_path=os.path.join(id_dir,"images")
#     masks_path=os.path.join(id_dir,"masks")
#     if not os.path.exists(images_path):
#         os.mkdir(images_path)
#         os.mkdir(masks_path)
#     print("copy file:{}".format(parent_id))
#     print(val_images[i],os.path.join(id_dir,image_name))
#     print(val_masks[i],os.path.join(id_dir,os.path.basename(val_masks[i])))
#     shutil.copy2(val_images[i],os.path.join(id_dir,image_name))
#     shutil.copy2(val_masks[i],os.path.join(id_dir,os.path.basename(val_masks[i])))





from PIL import Image
#  trains train
person_ids=[os.path.join(train_path,i) for i in os.listdir(train_path)]
upper=200
lower=-200
for person in person_ids:
    files=os.listdir(person)
    print(files)
    #files=['images', 'masks', 'segmentation-106.nii', 'volume-106.nii']
    for file in files:
        if "vol" in file:
            image_file=file
        elif "seg" in file:
            mask_file=file
    print(image_file,mask_file)
    image=sitk.ReadImage(os.path.join(person,image_file),sitk.sitkFloat32)
    mask=sitk.ReadImage(os.path.join(person,mask_file),sitk.sitkInt8)
    image = sitk.GetArrayFromImage(image)
    mask = sitk.GetArrayFromImage(mask)
    n,h,w=image.shape
    fusion_mask=mask.copy()
    fusion_mask[fusion_mask>0]=1
    # image[image > upper] = upper
    # image[image < lower] = lower
    z = np.any(fusion_mask, axis=(1, 2))
    start_slice, end_slice = np.where(z)[0][[0, -1]]
    print(start_slice,end_slice)
    for i in range(start_slice,end_slice,1):
        # 512*512
        sub_image=image[i].copy()
        sub_image[sub_image>1000]=1000
        #sub_image = hu_to_grayscale(sub_image)
        #sub_image=sub_image*255
        sub_image=Image.fromarray(sub_image).convert('L')
        sub_image.save(os.path.join(person,"images","image_{}.png".format(i)))
        sub_mask=mask[i]
        sub_mask[sub_mask==1]=128
        sub_mask[sub_mask==2]=255
        sub_mask=Image.fromarray(sub_mask).convert('L')
        sub_mask.save(os.path.join(person, "masks", "mask_{}.png".format(i)))
        # sub_image=image[i].astype(np.int8)
        # cv2.imshow(sub_image)
        # cv2





