import torch
import torch.nn as  nn
import argparse
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from torchvision import transforms as T
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import os
import cv2
from datasets import get_dataloder
import time
import tqdm

from models import get_models,models_dict





# Validation and visualization
Image_dir=r'E:\datasets\isic2018\valid'
Mask_dir=r'E:\datasets\isic2018\valid_GT'


parse=argparse.ArgumentParser("BaseLine Model Inference !")
parse.add_argument("--model",type=str,default='')
parse.add_argument('--model_weight1',type=str,default='')
parse.add_argument("--model2",type=str,default='')
parse.add_argument('--model_weight2',type=str,default='')


parse.add_argument('--image',type=str,default=Image_dir)
parse.add_argument('--mask',type=str,default=Mask_dir)
parse.add_argument('--im_channel', type=int, default=3, help="input image channel ")
parse.add_argument('--class_num', type=int, default=1, help="output feature channel")
parse.add_argument('--init_weight_type', type=str, choices=["kaiming", 'normal', 'xavier', 'orthogonal'],
                    default="kaiming", help=" model init mode")
parse.add_argument('--deepsupervision', action='store_true', help=" deepsupervision for  unet++")
parse.add_argument('--time_step', type=int, default=3, help=" r2unet use time step !")
parse.add_argument('--alpha', type=float, default=1.67, help=" multires unet channel changg ")
args=parse.parse_args()


#
# def create_dir(root,dir_name):
#     path=os.path.join(root,dir_name)
#     if not os.path.exists(path):
#         os.mkdir(path)
#     return path
#
#
# res_dir=r'C:\Users\rileyliu\Desktop\images_res\isic'
# image_save_dir=create_dir(res_dir,"images")
# mask_save_dir=create_dir(res_dir,"masks")
# unet_dir=create_dir(res_dir,'unet')
# nas_search_dir=create_dir(res_dir,'nas_search_net')
#
#
# def show_images(images,masks,output1,output2,filename):
#     image = Image.fromarray(images)
#     mask = Image.fromarray(masks)
#     output = Image.fromarray(output1)
#     output2 = Image.fromarray(output2)
#     image.save(os.path.join(image_save_dir, '{}.png'.format(filename)))
#     mask.save(os.path.join(mask_save_dir, '{}.png'.format(filename)))
#     output.save(os.path.join(unet_dir, '{}.png'.format(filename)))
#     output2.save(os.path.join(nas_search_dir, '{}.png'.format(filename)))
#
#
#
# def isic_transform(image_dir,mask_dir):
#     '''
#     :param image: PIL.Image
#     :return:
#     '''
#     # _img=cv2.imread(image_dir,1)
#     # _target=cv2.imread(mask_dir,0)
#     # _img=cv2.resize(_img,(256,256),interpolation=cv2.INTER_LINEAR)
#     # _target=cv2.resize(_target,(256,256),interpolation=cv2.INTER_NEAREST)
#     _img = Image.open(image_dir).convert('RGB')
#     _target = Image.open(mask_dir)
#     _img = _img.resize((256, 256), Image.BILINEAR)
#     _target = _target.resize((256, 256), Image.NEAREST)
#     img = tf.to_tensor(_img)
#     img = tf.normalize(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)).unsqueeze(0)
#     flipimg=np.asarray(_img)
#     flipimg=np.fliplr(flipimg)
#     flipimg=Image.fromarray(flipimg)
#     flipimg = tf.to_tensor(flipimg)
#     flipimg = tf.normalize(flipimg, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)).unsqueeze(0)
#     return img,flipimg,_img,_target
#
#
# def inference_isic(model1,model2,img_dir,mask_dir):
#     model1.eval()
#     model2.eval()
#     filenames = os.listdir(img_dir)
#     data_list = []
#     gt_list = []
#     img_ids = []
#     for filename in filenames:
#         ext = os.path.splitext(filename)[-1]
#         if ext == '.jpg':
#             filename = filename.split('_')[-1][:-len('.jpg')]
#             img_ids.append(filename)
#             data_list.append('ISIC_' + filename + '.jpg')
#             gt_list.append('ISIC_' + filename + '_segmentation.png')
#
#     assert (len(data_list) == len(gt_list))
#     data_list = [os.path.join(img_dir, i) for i in data_list]
#     gt_list = [os.path.join(mask_dir, i) for i in gt_list]
#
#
#     hard_filenames=['18','24','26','31','42','49'
#                     '56','62','73','81','91','97','113','153','184','16071','246','288','311','319',
#                     '324','358','387','393','395','499','504','520','529',531,547,549,1140,1148,
#                     1152,1184,1442,2829,3346,4115,5555,6612,6914,7557,8913,9873,9875,9934,10093,
#                     11107,11110,11168,11349,12090,12136,12149,12167,12187,12212,12216,12290,
#                     12329,12512,12516,12713,12773,12876,12999,13000,13010,13063,13120,13164,13227,
#                     13242,13393,13493,13516,13518,13549,13709,13813,13832,13988,14132,14189,14221,14639,
#                     14693,14912,15102,15176,15237,15330,155417,15443,16068]
#
#     better_filenames=['16','63','75','101','105','131','148','164','184','198','252','276','330',
#                       '397','433','458','476','480',1119,1212,1262,1306,1374,3346,6671,9504,
#                       9895,9992,10041,10044,10175,10183,10213,10382,10452,10456,11079,11130,11159,
#                       12318,12495,12897,12961,13146,13340,13371,13411,13807,13910,13918,14090,14693,
#                       14697,14850,14898,14904,15062,15166,15207,15483,15563,]
#
#     easy_filenames=['34','39','52','57','117','164','165','182','207','213','222','225','232']
#
#
#     dataset_wrong_case=[9800,9934,9951,10021,10361,10584,11227,13310,13600,13673,13680,15132,15152,15251,
#                         16036,]
#
#     all_bad_case=[10320,10361,10445,10457,10477,11081,11084,11121,12369,12484,12726,12740,12768,
#                   12786,12789,12876,12877,13120,13310,13393,13552,13832,13975,14222,14328,14372,14385,
#                   14434,14454,14480,14503,14506,14580,14628,14786,14931,14932,14963,14982,14985,15020,
#                   15021,15062,15309,15537,15947,15966,15969,15983,156008,16034,16037,16058,16068,]
#
#     for i in range(len(data_list)):
#         file_name=img_ids[i]
#         print("Filename:{}".format(file_name))
#         img,flipimg,original_img,mask=isic_transform(data_list[i],gt_list[i])
#         output=model1(img)
#         #flip_output=model1(flipimg)
#         output=torch.sigmoid(output).data.cpu().numpy()[0,0,:,:]
#         output2=model2(img)
#         output2=torch.sigmoid(output2[-1]).data.cpu().numpy()[0,0,:,:]
#         # flip_output=torch.sigmoid(flip_output).data.cpu().numpy()[0,0,:,:]
#         # flip_output=np.fliplr(flip_output)
#         # 可视化
#         oimage=np.asarray(original_img).astype(np.uint8)
#         mask=np.asarray(mask).astype(np.uint8)
#         output=(output>0.5).astype(np.uint8)
#         output2=(output2>0.5).astype(np.uint8)
#         # flip_output=(flip_output>0.5).astype(np.uint8)
#         mask[mask>=1]=255
#         output[output>=1]=255
#         output2[output2>=1]=255
#         # flip_output[flip_output>=1]=255
#
#         #rgb
#         #img[..., 2] = np.where(mask == 1, 255, img[..., 2])
#         contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(oimage, contours, -1, (0, 0, 255), 1,lineType=cv2.LINE_AA)
#
#         output_contours, _ = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(oimage, output_contours, -1, (0, 255,0), 1,lineType=cv2.LINE_AA)
#
#         output_contours, _ = cv2.findContours(output2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(oimage, output_contours, -1, (255, 0,0), 1,lineType=cv2.LINE_AA)
#
#         show_images(oimage.copy(),mask.copy(),output.copy(),output2.copy(),file_name)
#         # cv2.imwrite(os.path.join(image_save_dir,'{}.png'.format(filename)),oimage)
#
#         # flip_output_contours, _ = cv2.findContours(flip_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         # cv2.drawContours(oimage, flip_output_contours, -1, (255, 0,0), 1,lineType=cv2.LINE_AA)
#         # oimage=oimage[:,:,[2,1,0]]
#         # cv2.imread(os.path.join())
#         # cv2.imshow('original_mask',oimage)
#         # cv2.imshow('mask',mask)
#         # cv2.imshow('output',output)
#         # cv2.imshow('output2',output2)
#         # # cv2.imshow('flip_output',flip_output)
#         # cv2.waitKey()
#         # # [75,101]
#         # # 容易 [78,107，129]
#



def inference_time_test(args,model):
    # filenames = os.listdir(images)
    # data_list = []
    # gt_list = []
    # img_ids = []
    # for filename in filenames:
    #     ext = os.path.splitext(filename)[-1]
    #     if ext == '.jpg':
    #         filename = filename.split('_')[-1][:-len('.jpg')]
    #         img_ids.append(filename)
    #         data_list.append('ISIC_' + filename + '.jpg')
    #         gt_list.append('ISIC_' + filename + '_segmentation.png')
    #
    # assert (len(data_list) == len(gt_list))
    # data_list = [os.path.join(images, i) for i in data_list]
    # gt_list = [os.path.join(mask, i) for i in gt_list]

    dataloader=get_dataloder(args,split_flag='train')
    count=0
    model=model.to(args.device)
    model.eval()
    start=time.time()
    for step, (input, target,_) in tqdm.tqdm(enumerate(dataloader)):
        input = input.to(args.device)
        target = target.to(args.device)
        output=model(input)
        count+=args.train_batch
        if step>20:
            break
    end_tim=time.time()
    return (end_tim-start)/count





if __name__=="__main__":
    # main(args)
    from utils import save_checkpoint, calc_parameters_count, get_logger, get_gpus_memory_info,get_model_complexity_info
    args.device=torch.device('cpu')
    args.dataset='isic2018'
    args.train_batch=1
    args.val_batch=1
    args.num_workers=2
    args.crop_size=256
    args.base_size=256

    model_list=['unet','attention_unet_v1','unet++','r2unet','multires_unet']

    for model_name in model_list:
        args.model=model_name
        model=get_models(args)
        # for name,moudle in model.named_modules():
        #     print(name,type(moudle))
        flop, param = get_model_complexity_info(model, (3, 192, 256), as_strings=True, print_per_layer_stat=False)
        print("GFLOPs: {}".format(flop))
        print("Params: {}".format(param))
        #
        # avg_time=inference_time_test(args,model)
        # print("name:{} time:{}".format(model_name,avg_time))






