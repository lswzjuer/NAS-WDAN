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


import genotypes
from models import get_models,models_dict
from nas_search_unet_prune import BuildNasUnetPrune
from utils import BinaryIndicatorsMetric

# Validation and visualization
Image_dir=r'E:\datasets\isic2018\valid'
Mask_dir=r'E:\datasets\isic2018\valid_GT'
# model
Model_name='unet'
Model_dir1='./logs/logs_coslr/unet/isic2018/20200229-035150/checkpoint.pth.tar'
model_dir2='./nas_search_unet/logs/isic2018/prune_20200313-063406_32_32_ep300_double_deep/model_best.pth.tar'

parse=argparse.ArgumentParser("BaseLine Model Inference !")
parse.add_argument("--model",type=str,default=Model_name)
parse.add_argument('--model_weight1',type=str,default=Model_dir1)
parse.add_argument("--model2",type=str,default='nas_unet_stage0_layer9_test')
parse.add_argument('--model_weight2',type=str,default=model_dir2)


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



def create_dir(root,dir_name):
    path=os.path.join(root,dir_name)
    if not os.path.exists(path):
        os.mkdir(path)
    return path


res_dir=r'C:\Users\rileyliu\Desktop\images_res\isic'
image_save_dir=create_dir(res_dir,"images")
mask_save_dir=create_dir(res_dir,"masks")
unet_dir=create_dir(res_dir,'unet')
nas_search_dir=create_dir(res_dir,'nas_search_net')


def show_images(images,masks,output1,output2,filename):
    image = Image.fromarray(images)
    mask = Image.fromarray(masks)
    output = Image.fromarray(output1)
    output2 = Image.fromarray(output2)
    image.save(os.path.join(image_save_dir, '{}.png'.format(filename)))
    mask.save(os.path.join(mask_save_dir, '{}.png'.format(filename)))
    output.save(os.path.join(unet_dir, '{}.png'.format(filename)))
    output2.save(os.path.join(nas_search_dir, '{}.png'.format(filename)))



def isic_transform(image_dir,mask_dir):
    '''
    :param image: PIL.Image
    :return:
    '''
    # _img=cv2.imread(image_dir,1)
    # _target=cv2.imread(mask_dir,0)
    # _img=cv2.resize(_img,(256,256),interpolation=cv2.INTER_LINEAR)
    # _target=cv2.resize(_target,(256,256),interpolation=cv2.INTER_NEAREST)
    _img = Image.open(image_dir).convert('RGB')
    _target = Image.open(mask_dir)
    _img = _img.resize((256, 256), Image.BILINEAR)
    _target = _target.resize((256, 256), Image.NEAREST)
    img = tf.to_tensor(_img)
    img = tf.normalize(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)).unsqueeze(0)
    flipimg=np.asarray(_img)
    flipimg=np.fliplr(flipimg)
    flipimg=Image.fromarray(flipimg)
    flipimg = tf.to_tensor(flipimg)
    flipimg = tf.normalize(flipimg, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)).unsqueeze(0)
    return img,flipimg,_img,_target


def inference_isic(model1,model2,img_dir,mask_dir):
    OtherVal = BinaryIndicatorsMetric()
    model1.eval()
    model2.eval()
    filenames = os.listdir(img_dir)
    data_list = []
    gt_list = []
    img_ids = []
    filenames=sorted(filenames,key=lambda x:int(x.split('_')[-1][:-len('.jpg')]))
    for filename in filenames:
        ext = os.path.splitext(filename)[-1]
        if ext == '.jpg':
            filename = filename.split('_')[-1][:-len('.jpg')]
            img_ids.append(filename)
            data_list.append('ISIC_' + filename + '.jpg')
            gt_list.append('ISIC_' + filename + '_segmentation.png')

    assert (len(data_list) == len(gt_list))
    data_list = [os.path.join(img_dir, i) for i in data_list]
    gt_list = [os.path.join(mask_dir, i) for i in gt_list]


    hard_filenames=['18','24','26','31','42','49'
                    '56','62','73','81','91','97','113','153','184','16071','246','288','311','319',
                    '324','358','387','393','395','499','504','520','529',531,547,549,1140,1148,
                    1152,1184,1442,2829,3346,4115,5555,6612,6914,7557,8913,9873,9875,9934,10093,
                    11107,11110,11168,11349,12090,12136,12149,12167,12187,12212,12216,12290,
                    12329,12512,12516,12713,12773,12876,12999,13000,13010,13063,13120,13164,13227,
                    13242,13393,13493,13516,13518,13549,13709,13813,13832,13988,14132,14189,14221,14639,
                    14693,14912,15102,15176,15237,15330,155417,15443,16068]

    better_filenames=['16','63','75','101','105','131','148','164','184','198','252','276','330',
                      '397','433','458','476','480',1119,1212,1262,1306,1374,3346,6671,9504,
                      9895,9992,10041,10044,10175,10183,10213,10382,10452,10456,11079,11130,11159,
                      12318,12495,12897,12961,13146,13340,13371,13411,13807,13910,13918,14090,14693,
                      14697,14850,14898,14904,15062,15166,15207,15483,15563,]

    easy_filenames=['34','39','52','57','117','164','165','182','207','213','222','225','232']


    dataset_wrong_case=[9800,9934,9951,10021,10361,10584,11227,13310,13600,13673,13680,15132,15152,15251,
                        16036,]

    all_bad_case=[10320,10361,10445,10457,10477,11081,11084,11121,12369,12484,12726,12740,12768,
                  12786,12789,12876,12877,13120,13310,13393,13552,13832,13975,14222,14328,14372,14385,
                  14434,14454,14480,14503,14506,14580,14628,14786,14931,14932,14963,14982,14985,15020,
                  15021,15062,15309,15537,15947,15966,15969,15983,156008,16034,16037,16058,16068,]

    for i in range(len(data_list)):
        file_name=img_ids[i]
        print("Filename:{}".format(file_name))
        img,flipimg,original_img,mask=isic_transform(data_list[i],gt_list[i])
        output=model1(img)
        #flip_output=model1(flipimg)
        #output=torch.sigmoid(output).data.cpu().numpy()[0,0,:,:]
        output2=model2(img)

        nas_output = output[-1].clone()
        nas_output = nas_output.view(nas_output.size(0), -1)
        target = torch.from_numpy(np.asarray(mask))
        target = target.unsqueeze(0).unsqueeze(0)
        target = target.view(target.size(0), -1)
        OtherVal.update(labels=target, preds=nas_output, n=1)
        # OtherVal.update(labels=target, preds=outputs_original[0].view(outputs_original[0].size(0), -1), n=1)
        #
        # output2=torch.sigmoid(output2[-1]).data.cpu().numpy()[0,0,:,:]
        # # flip_output=torch.sigmoid(flip_output).data.cpu().numpy()[0,0,:,:]
        # # flip_output=np.fliplr(flip_output)
        # # 可视化
        # oimage=np.asarray(original_img).astype(np.uint8)
        # mask=np.asarray(mask).astype(np.uint8)
        # output=(output>0.5).astype(np.uint8)
        # output2=(output2>0.5).astype(np.uint8)
        # # flip_output=(flip_output>0.5).astype(np.uint8)
        # mask[mask>=1]=255
        # output[output>=1]=255
        # output2[output2>=1]=255
        # # flip_output[flip_output>=1]=255
        #
        # #rgb
        # #img[..., 2] = np.where(mask == 1, 255, img[..., 2])
        # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(oimage, contours, -1, (0, 0, 255), 1,lineType=cv2.LINE_AA)
        #
        # output_contours, _ = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(oimage, output_contours, -1, (0, 255,0), 1,lineType=cv2.LINE_AA)
        #
        # output_contours, _ = cv2.findContours(output2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(oimage, output_contours, -1, (255, 0,0), 1,lineType=cv2.LINE_AA)
        #
        # show_images(oimage.copy(),mask.copy(),output.copy(),output2.copy(),file_name)
        # cv2.imwrite(os.path.join(image_save_dir,'{}.png'.format(filename)),oimage)

        # flip_output_contours, _ = cv2.findContours(flip_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(oimage, flip_output_contours, -1, (255, 0,0), 1,lineType=cv2.LINE_AA)
        # oimage=oimage[:,:,[2,1,0]]
        # cv2.imread(os.path.join())
        # cv2.imshow('original_mask',oimage)
        # cv2.imshow('mask',mask)
        # cv2.imshow('output',output)
        # cv2.imshow('output2',output2)
        # # cv2.imshow('flip_output',flip_output)
        # cv2.waitKey()
        # # [75,101]
        # # 容易 [78,107，129]
    value=OtherVal.get_avg
    mr, ms, mp, mf, mjc, md, macc = value
    print("Acc:{:.3f} Dice:{:.3f} Jc:{:.3f}".format(macc, md, mjc))


def main(args):
    model1=get_models(args)
    ckpt1=torch.load(args.model_weight1,map_location='cpu')
    model1.load_state_dict(ckpt1['state_dict'])
    # inference_isic(model,args.image,args.mask)

    ckpt2=torch.load(args.model_weight2,map_location='cpu')
    genotype = eval('genotypes.%s' % 'stage1_layer9_110epoch_double_deep_final')
    #BuildNasUnetPrune
    model2=BuildNasUnetPrune(
        genotype=genotype,
        input_c=3,
        c=16,
        num_classes=1,
        meta_node_num=4,
        layers=9,
        dp=0,
        use_sharing=True,
        double_down_channel=True,
        aux=True,
    )
    model2.load_state_dict(ckpt2['state_dict'])
    inference_isic(model1,model2,args.image,args.mask)
    # for key,value in ckpt2['state_dict'].items():
    #     print(key)



if __name__=="__main__":
    main(args)


# def regist_hook_outfeature(self, model):
#     """
#     创建hook获取每层的结果
#     """
#     out_feat = OrderedDict()
#     hooks = []
#     # print('layer num:', self.layers_num)
#     all_op_type = self._all_op_type
#
#     def _make_hook(m):
#
#         def _hook(m, input, output):
#             class_name = str(m.__class__).split(".")[-1].split("'")[0]
#             layer_type = type(m).__name__
#             idx = len(out_feat) % (int(self.layers_num + 1))
#             if (idx == 0):
#                 out_feat.clear()
#                 out_feat['image'] = input[0].detach().cpu().numpy()
#                 idx = len(out_feat)
#             name_keys = "%s_%i" % (class_name, idx)
#             # name_keys = "%s_%i" % (class_name, len(out_feat))
#             out_feat[name_keys] = output.detach().cpu().numpy()
#
#         if (type(m).__name__ in all_op_type):
#             hooks.append(m.register_forward_hook(_hook))
#
#     # 对模型中的每个模块的输出均执行_make_hook操作，本质商也就是保存每一层的输出特征图
#     # 后面我们使用的时候，要有针对性的修改，具体来说就是只保存有意义的特征图的修改
#
#     model.apply(_make_hook)
#     # register hook
#
#     return out_feat, hooks


