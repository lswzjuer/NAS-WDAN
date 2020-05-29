import os
import time
import argparse
from tqdm import tqdm
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import sys
from PIL import Image

from datasets import get_dataloder,datasets_dict
from models import get_models,models_dict
from utils import save_checkpoint,calc_parameters_count,get_logger,get_gpus_memory_info
from utils import RefugeIndicatorsMetricBinary,BinaryIndicatorsMetric,AverageMeter
from utils import BCEDiceLoss,SoftDiceLoss,DiceLoss,BCEDiceLoss
from utils import MultiClassEntropyDiceLoss



def main(args):
    #################### init logger ###################################
    # args.model_list=["unet","unet++_deep","unet++_nodeep",'attention_unet_v1','multires_unet','r2unet_t3',
    #                  'unet_ep800dice','unet++_deep_ep800dice','unet++_nodeep_ep800dice','attention_unet_v1_ep800dice','multires_unet_ep800dice'
    #                  ]
    args.model_list=['unet','unet++',"attention_unet","multires_unet"]


    for model_name in args.model_list:
        if model_name=='unet':
            args.model='unet'
            model_weight_path='./logs/chaos/unet_ep150_v2/20200403-134703/checkpoint.pth.tar'
            model=get_models(args)
            model.load_state_dict(torch.load(model_weight_path, map_location='cpu')['state_dict'])
        elif model_name=='unet++':
            args.model='unet++'
            args.deepsupervision=False
            model_weight_path='./logs/chaos/unet++_ep150_v2/20200403-135401/checkpoint.pth.tar'
            model=get_models(args)
            model.load_state_dict(torch.load(model_weight_path, map_location='cpu')['state_dict'])

        # elif model_name == 'unet++_deep':
        #     args.model = 'unet++'
        #     args.deepsupervision = True
        #     model_weight_path = './logs/unet++_deep_ep1600/cvc/20200312-143345/model_best.pth.tar'
        #     model = get_models(args)
        #     model.load_state_dict(torch.load(model_weight_path, map_location='cpu')['state_dict'])

        elif model_name == 'attention_unet':
            args.model = 'attention_unet_v1'
            args.deepsupervision = False
            model_weight_path = './logs/chaos/attention_unet_v1_ep150_v2/20200403-135445/checkpoint.pth.tar'
            model = get_models(args)
            model.load_state_dict(torch.load(model_weight_path, map_location='cpu')['state_dict'])

        elif model_name == 'multires_unet':
            args.model = 'multires_unet'
            args.deepsupervision = False
            model_weight_path = './logs/chaos/multires_unet_ep150_v2/20200403-135549/checkpoint.pth.tar'
            model = get_models(args)
            model.load_state_dict(torch.load(model_weight_path, map_location='cpu')['state_dict'])


        else:
            raise  NotImplementedError()


        assert os.path.exists(args.save)
        args.model_save_path=os.path.join(args.save,model_name)
        logger = get_logger(args.model_save_path)
        args.save_images= os.path.join(args.model_save_path,"images")
        if not os.path.exists(args.save_images):
            os.mkdir(args.save_images)
        if args.manualSeed is None:
            args.manualSeed = random.randint(1, 10000)
        np.random.seed(args.manualSeed)
        torch.manual_seed(args.manualSeed)
        args.use_cuda = args.gpus > 0 and torch.cuda.is_available()
        args.device = torch.device('cuda' if args.use_cuda else 'cpu')
        if args.use_cuda:
            torch.cuda.manual_seed(args.manualSeed)
            cudnn.benchmark = True
        val_loader = get_dataloder(args, split_flag="valid")
        setting = {k: v for k, v in args._get_kwargs()}
        logger.info(setting)
        logger.info('param size = %fMB', calc_parameters_count(model))


        # init loss
        if args.loss == 'bce':
            criterion = nn.BCELoss()
        elif args.loss == 'bcelog':
            criterion = nn.BCEWithLogitsLoss()
        elif args.loss == "dice":
            criterion = DiceLoss()
        elif args.loss == "softdice":
            criterion = SoftDiceLoss()
        elif args.loss == 'bcedice':
            criterion = BCEDiceLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        if args.use_cuda:
            logger.info("load model and criterion to gpu !")

        model = model.to(args.device)
        criterion = criterion.to(args.device)
        infer(args, model, criterion, val_loader,logger,args.save_images)



def infer(args, model, criterion, val_loader,logger,path):
    OtherVal_v1=RefugeIndicatorsMetricBinary()
    OtherVal_v2=BinaryIndicatorsMetric()
    tloss_r=AverageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target,name) in tqdm(enumerate(val_loader)):
            # input: n,1,h,w   n,1,h,w   name:n list
            input = input.to(args.device)
            target = target.to(args.device)
            pred = model(input)
            if args.deepsupervision:
                pred=pred[-1].clone()


            # save the mask
            file_masks=pred.clone()
            file_masks=torch.sigmoid(file_masks).data.cpu().numpy()
            n,c,h,w=file_masks.shape
            assert n==len(file_masks)
            for i in range(len(file_masks)):
                subdir=name[0][i]
                file_index=name[1][i]
                if not os.path.exists(os.path.join(path,subdir)):
                    os.mkdir(os.path.join(path,subdir))
                file_mask=(file_masks[i][0] > 0.5).astype(np.uint8)
                file_mask[file_mask >= 1] = 255
                file_mask=Image.fromarray(file_mask)
                file_mask.save(os.path.join(path,subdir,file_index+".png"))

            # batchsize=8
            OtherVal_v1.update(pred,target)
            #preds_list = [pred.view(pred.size(0), -1) for pred in preds_list]
            target = target.view(target.size(0), -1)
            v_loss = criterion(pred.view(pred.size(0), -1), target)
            v_loss = v_loss * n
            tloss_r.update(v_loss.item(), 1)
            OtherVal_v2.update(labels=target,preds=pred.view(pred.size(0), -1))

        per1 = OtherVal_v1.avg()
        mr, ms, mp, mf, mjc, md, macc = OtherVal_v2.get_avg
        logger.info(" V1     dice:{:.3f} iou:{:.3f} acc:{:.3f}  ".format(per1[0], per1[1], per1[2]))
        logger.info(" V2     dice:{:.3f} iou:{:.3f} acc:{:.3f}  ".format(md, mjc, macc))
        return tloss_r.avg, per1



if __name__ == '__main__':
    models_name=models_dict.keys()
    datasets_name=datasets_dict.keys()
    parser = argparse.ArgumentParser(description='Unet serieas baseline')
    # Add default argument
    parser.add_argument('--model',  type=str, default='unet',choices=models_name,
                        help='Model to train and evaluation')
    parser.add_argument('--note' ,type=str, default='_',
                        help='model note ')
    parser.add_argument('--save',type=str,default='./nas_search_unet/eval/chaos')
    parser.add_argument('--dataset',type=str, default='chaos',choices=datasets_name,
                        help='Model to train and evaluation')
    parser.add_argument('--base_size', type=int, default=256, help="resize base size")
    parser.add_argument('--crop_size', type=int, default=256, help="crop  size")
    parser.add_argument('--im_channel', type=int, default=1, help="input image channel ")
    parser.add_argument('--class_num', type=int, default=1, help="output feature channel")
    parser.add_argument('--epoch', type=int, default=200, help="epochs")
    parser.add_argument('--train_batch', type=int, default=1, help="train_batch")
    parser.add_argument('--val_batch', type=int, default=8, help="val_batch ")
    parser.add_argument('--num_workers', type=int, default=2, help="dataloader numworkers")
    parser.add_argument('--init_weight_type',type=str, choices=["kaiming",'normal','xavier','orthogonal'],
                        default="kaiming",help=" model init mode")
    parser.add_argument('--print_freq', type=int, default=100, help=" print freq (iteras) ")
    parser.add_argument('--gpus', type=int,default=1, help=" use cuda or not ")
    parser.add_argument('--grad_clip',type=int, default=5,help=" grid clip to ignore grad boom")
    parser.add_argument('--manualSeed', type=int, default=100, help=" manualSeed ")
    #model special
    parser.add_argument('--deepsupervision', action='store_true', help=" deepsupervision for  unet++")
    parser.add_argument('--time_step',type=int, default=3,help=" r2unet use time step !")
    parser.add_argument('--alpha', type=float, default=1.67, help=" multires unet channel changg ")
    # optimer
    parser.add_argument('--loss',type=str, choices=['bce','bcelog','dice','softdice','bcedice'],
                        default="bcedice",help="loss name ")
    args = parser.parse_args()
    main(args)