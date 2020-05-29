

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
Image_dir=r'E:\datasets\CVC-ClinicDB\valid'
Mask_dir=r'E:\datasets\CVC-ClinicDB\valid_GT'
# model

parse=argparse.ArgumentParser("BaseLine Model Inference !")
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


res_dir=r'C:\Users\rileyliu\Desktop\images_res\cvc'
model_name_list = ['unet', 'unet++', 'multires_unet', 'attention_unet_v1',"nas_search"]
create_dir(res_dir,'images')
create_dir(res_dir,'mask')
for name in model_name_list:
    create_dir(res_dir, name)


# def show_images(images,masks,o1,o2,o3,o4,filename):
#     image = Image.fromarray(images)
#     mask = Image.fromarray(masks)
#
#     unet = Image.fromarray(o1)
#     unetpp = Image.fromarray(o2)
#     multires_unet = Image.fromarray(o3)
#     attention_unet_v1 = Image.fromarray(o4)
#
#     image.save(os.path.join(os.path.join(res_dir,'images'), '{}.png'.format(filename)))
#     mask.save(os.path.join(os.path.join(res_dir,'mask'), '{}.png'.format(filename)))
#     unet.save(os.path.join(os.path.join(res_dir,'unet'), '{}.png'.format(filename)))
#     unetpp.save(os.path.join(os.path.join(res_dir,'unet++'), '{}.png'.format(filename)))
#     multires_unet.save(os.path.join(os.path.join(res_dir,'multires_unet'), '{}.png'.format(filename)))
#     attention_unet_v1.save(os.path.join(os.path.join(res_dir,'attention_unet_v1'), '{}.png'.format(filename)))


def show_images(images,masks,o1,o2,filename):
    image = Image.fromarray(images)
    mask = Image.fromarray(masks)

    unet = Image.fromarray(o1)
    nas_search = Image.fromarray(o2)


    image.save(os.path.join(os.path.join(res_dir,'images'), '{}.png'.format(filename)))
    mask.save(os.path.join(os.path.join(res_dir,'mask'), '{}.png'.format(filename)))
    unet.save(os.path.join(os.path.join(res_dir,'unet'), '{}.png'.format(filename)))
    nas_search.save(os.path.join(os.path.join(res_dir,'nas_search'), '{}.png'.format(filename)))



def isic_transform(image_dir,mask_dir):
    '''
    :param image: PIL.Image
    :return:
    '''
    # _img=cv2.imread(image_dir,1)
    # _target=cv2.imread(mask_dir,0)
    # _img=cv2.resize(_img,(256,256),interpolation=cv2.INTER_LINEAR)
    # _target=cv2.resize(_target,(256,256),interpolation=cv2.INTER_NEAREST)

    _img = cv2.imread(image_dir, cv2.IMREAD_COLOR)
    _img = _img[:, :, [2, 1, 0]]
    _img = Image.fromarray(_img).convert("RGB")
    _target = Image.open(mask_dir)
    _img = _img.resize((256, 192), Image.BILINEAR)
    _target = _target.resize((256, 192), Image.NEAREST)
    img = tf.to_tensor(_img)
    img = tf.normalize(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)).unsqueeze(0)
    return img,_img,_target



def inference_isic(models_list,img_dir,mask_dir):
    OtherVal = BinaryIndicatorsMetric()
    for model in models_list:
        model.eval()
    filenames = os.listdir(img_dir)
    data_list = []
    gt_list = []
    img_ids = []
    for filename in filenames:
        data_list.append(filename)
        gt_list.append(filename)
        img_ids.append(filename)
        assert os.path.splitext(filename)[-1] == '.tif'

    assert (len(data_list) == len(gt_list))
    data_list = [os.path.join(img_dir, i) for i in data_list]
    gt_list = [os.path.join(mask_dir, i) for i in gt_list]


    hard_filenames=[]

    better_filenames=[]

    easy_filenames=[]

    dataset_wrong_case=[]

    all_bad_case=[]

    model_name_list=['unet','unet++','multires_unet','attention_unet_v1','nas_search']
    for i in range(len(data_list)):
        file_name=img_ids[i].split('.')[0]
        print("Filename:{}".format(file_name))
        img,original_img,mask=isic_transform(data_list[i],gt_list[i])
        outputs_original=[model(img) for model in models_list]
        nas_output=outputs_original[-1][-1].clone()
        nas_output=nas_output.view(nas_output.size(0),-1)
        target=torch.from_numpy(np.asarray(mask))
        target=target.unsqueeze(0).unsqueeze(0)
        target=target.view(target.size(0), -1)
        # OtherVal.update(labels=target, preds=nas_output, n=1)
        OtherVal.update(labels=target, preds=outputs_original[0].view(outputs_original[0].size(0),-1), n=1)

        # outputs=[]
        # for index,output in enumerate(outputs_original):
        #     if isinstance(output,list):
        #         print("Index:{} is nas search mmodel !".format(index))
        #         outputs.append(torch.sigmoid(output[-1]).data.cpu().numpy()[0,0,:,:])
        #     else:
        #         outputs.append(torch.sigmoid(output).data.cpu().numpy()[0,0,:,:])
        # outputs=[(output>0.5).astype(np.uint8) for output in outputs]
        # for output in outputs:
        #     output[output>=1]=255
        # #flip_output=model1(flipimg)
        # # flip_output=torch.sigmoid(flip_output).data.cpu().numpy()[0,0,:,:]
        # # flip_output=np.fliplr(flip_output)
        # # 可视化
        # oimage=np.asarray(original_img).astype(np.uint8)
        # mask=np.asarray(mask).astype(np.uint8)
        # # flip_output=(flip_output>0.5).astype(np.uint8)
        # mask[mask>=1]=255
        #
        # # flip_output[flip_output>=1]=255
        # #img[..., 2] = np.where(mask == 1, 255, img[..., 2])
        # unet=outputs[0]
        # # unetpp=outputs[1]
        # # multires_unet=outputs[2]
        # # attention_unet_v1=outputs[3]
        # nas_search_output=outputs[-1]
        #
        # # rgb
        # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(oimage, contours, -1, (0, 0,255), 1,lineType=cv2.LINE_AA)
        # output_contours, _ = cv2.findContours(unet, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(oimage, output_contours, -1, (0, 255,0), 1,lineType=cv2.LINE_AA)
        #
        # # output_contours, _ = cv2.findContours(unetpp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # # cv2.drawContours(oimage, output_contours, -1, (0, 0,255), 1,lineType=cv2.LINE_AA)
        #
        # # output_contours, _ = cv2.findContours(multires_unet, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # # cv2.drawContours(oimage, output_contours, -1, (0, 255,255), 1,lineType=cv2.LINE_AA)
        # #
        # #
        # # output_contours, _ = cv2.findContours(attention_unet_v1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # # cv2.drawContours(oimage, output_contours, -1, (255, 0,255), 1,lineType=cv2.LINE_AA)
        #
        #
        # output_contours, _ = cv2.findContours(nas_search_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(oimage, output_contours, -1, (255, 0,0), 1,lineType=cv2.LINE_AA)
        #
        # # #show_images(oimage.copy(),mask.copy(),unet.copy(),unetpp.copy(),multires_unet.copy(),attention_unet_v1.copy(),file_name)
        # show_images(oimage.copy(), mask.copy(), unet.copy(),nas_search_output.copy(), file_name)
    value=OtherVal.get_avg
    mr, ms, mp, mf, mjc, md, macc = value
    print("Acc:{:.3f} Dice:{:.3f} Jc:{:.3f}".format(macc, md, mjc))



def main(args):
    # 0.762
    # args.model='unet'
    # model1=get_models(args)
    # model1.load_state_dict(torch.load(r'E:\segmentation\Image_Segmentation\logs\cvc_logs\unet_ep1600\cvc\20200312-143050\model_best.pth.tar',map_location='cpu')['state_dict'])

    # # 0.766/0.773
    # args.model='unet++'
    # model2=get_models(args)
    # model2.load_state_dict(torch.load(r'E:\segmentation\Image_Segmentation\logs\cvc_logs\unet++_nodeep_ep800\cvc\no_deep\model_best.pth.tar',map_location='cpu')['state_dict'])
    #
    # # mutilres 0.695
    # args.model='multires_unet'
    # model3=get_models(args)
    # model3.load_state_dict(torch.load(r'E:\segmentation\Image_Segmentation\logs\cvc_logs\multires_unet_800\cvc\20200310-172036\checkpoint.pth.tar',map_location='cpu')['state_dict'])
    #
    #
    # attention_unet 0.778
    args.model = 'attention_unet_v1'
    model4 = get_models(args)
    model4.load_state_dict(torch.load(r'E:\segmentation\Image_Segmentation\logs\cvc_logs\attention_unet_v1_ep1600\cvc\20200312-143413\model_best.pth.tar',map_location='cpu')['state_dict'])

    genotype = eval('genotypes.%s' % 'layer7_double_deep')
    #BuildNasUnetPrune
    model5=BuildNasUnetPrune(
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
    model5.load_state_dict(torch.load(r'E:\segmentation\Image_Segmentation\nas_search_unet\logs\cvc\layer7_double_deep_ep1600_20200320-200539\model_best.pth.tar',map_location='cpu')['state_dict'])
    models_list=[model4,model5]
    inference_isic(models_list,args.image,args.mask)


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


