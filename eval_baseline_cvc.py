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
from utils import BinaryIndicatorsMetric,AverageMeter
from utils import BCEDiceLoss,SoftDiceLoss,DiceLoss,BCEDiceLoss



def main(args):
    #################### init logger ###################################
    args.model_list=["unet","unet++",'attention_unet_v1','multires_unet','r2unet_t3']


    for model_name in args.model_list:
        if model_name=='unet':
            args.model='unet'
            model_weight_path='./logs/unet_ep1600/cvc/20200312-143050/model_best.pth.tar'
            model=get_models(args)
            model.load_state_dict(torch.load(model_weight_path, map_location='cpu')['state_dict'])
        elif model_name=='unet++':
            args.model='unet++'
            args.deepsupervision=False
            model_weight_path='./logs/unet++_ep1600/cvc/20200312-143358/model_best.pth.tar'
            model=get_models(args)
            model.load_state_dict(torch.load(model_weight_path, map_location='cpu')['state_dict'])

        elif model_name == 'attention_unet_v1':
            args.model = 'attention_unet_v1'
            model_weight_path = './logs/attention_unet_v1_ep1600/cvc/20200312-143413/model_best.pth.tar'
            model = get_models(args)
            model.load_state_dict(torch.load(model_weight_path, map_location='cpu')['state_dict'])

        elif model_name == 'multires_unet':
            args.model = 'multires_unet'
            model_weight_path = './logs/multires_unet_ep1600_t2/20200322-194117/model_best.pth.tar'
            model = get_models(args)
            model.load_state_dict(torch.load(model_weight_path, map_location='cpu')['state_dict'])

        # change bn relu order
        elif model_name == 'multires_unet_align':
            args.model = 'multires_unet'
            model_weight_path = './logs/multires_unet_ep1600_chbnrelu/20200327-184457/model_best.pth.tar'
            model = get_models(args)
            model.load_state_dict(torch.load(model_weight_path, map_location='cpu')['state_dict'])


        elif model_name == 'r2unet_t3':
            args.model = 'r2unet'
            args.time_step=3
            model_weight_path = './logs/r2unet_ep1600_t2/20200324-032815/model_best.pth.tar'
            model = get_models(args)
            model.load_state_dict(torch.load(model_weight_path, map_location='cpu')['state_dict'])


        elif model_name == 'unet_ep800dice':
            args.model = 'unet'
            model_weight_path = './logs/unet_ep800_bcedice/cvc/20200315-043021/model_best.pth.tar'
            model = get_models(args)
            model.load_state_dict(torch.load(model_weight_path, map_location='cpu')['state_dict'])

        elif model_name=='unet++_nodeep_ep800dice':
            args.model='unet++'
            args.deepsupervision=False
            model_weight_path='./logs/unet++_ep800_bcedice/cvc/20200315-043214/model_best.pth.tar'
            model=get_models(args)
            model.load_state_dict(torch.load(model_weight_path, map_location='cpu')['state_dict'])
        elif model_name == 'unet++_deep_ep800dice':
            args.model = 'unet++'
            args.deepsupervision = True
            model_weight_path = './logs/unet++_deep_ep800_bcedice/cvc/20200315-043134/model_best.pth.tar'
            model = get_models(args)
            model.load_state_dict(torch.load(model_weight_path, map_location='cpu')['state_dict'])

        elif model_name == 'attention_unet_v1_ep800dice':
            args.model = 'attention_unet_v1'
            args.deepsupervision=False
            model_weight_path = './logs/attention_unet_v1_ep800_bcedice/cvc/20200315-043300/model_best.pth.tar'
            model = get_models(args)
            model.load_state_dict(torch.load(model_weight_path, map_location='cpu')['state_dict'])

        elif model_name == 'multires_unet_ep800dice':
            args.model = 'multires_unet'
            args.deepsupervision=False
            model_weight_path = './logs/multires_unet_ep800_bcedice/cvc/20200312-173031/model_best.pth.tar'
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
    OtherVal = BinaryIndicatorsMetric()
    val_loss = AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target,name) in tqdm(enumerate(val_loader)):
            input = input.to(args.device)
            target = target.to(args.device)
            pred = model(input)
            if args.deepsupervision:
                pred=pred[-1].clone()
            # sabe predit mask
            # save the mask
            file_masks=pred.clone()
            file_masks=torch.sigmoid(file_masks).data.cpu().numpy()
            n,c,h,w=file_masks.shape
            assert n==len(file_masks)
            for i in range(len(file_masks)):
                file_index=int(name[i].split('.')[0])
                file_mask=(file_masks[i][0] > 0.5).astype(np.uint8)
                file_mask[file_mask >= 1] = 255
                file_mask=Image.fromarray(file_mask)
                file_mask.save(os.path.join(path,str(file_index)+".png"))

            # compute loss
            pred = pred.view(pred.size(0), -1)
            target = target.view(target.size(0), -1)
            v_loss = criterion(pred, target)
            val_loss.update(v_loss.item(), 1)
            OtherVal.update(labels=target, preds=pred, n=1)
        vmr, vms, vmp, vmf, vmjc, vmd, vmacc = OtherVal.get_avg
        # mvmr, mvms, mvmp, mvmf, mvmjc, mvmd, mvmacc = valuev2
        logger.info("Val_Loss:{:.5f} Acc:{:.5f} Dice:{:.5f} Jc:{:.5f}".format(val_loss.avg, vmacc, vmd, vmjc))


if __name__ == '__main__':
    models_name=models_dict.keys()
    datasets_name=datasets_dict.keys()
    parser = argparse.ArgumentParser(description='Unet serieas baseline')
    # Add default argument
    parser.add_argument('--model',  type=str, default='unet',choices=models_name,
                        help='Model to train and evaluation')
    parser.add_argument('--note' ,type=str, default='_',
                        help='model note ')
    parser.add_argument('--save',type=str,default='./nas_search_unet/eval/cvc')
    parser.add_argument('--dataset',type=str, default='cvc',choices=datasets_name,
                        help='Model to train and evaluation')
    parser.add_argument('--base_size', type=int, default=256, help="resize base size")
    parser.add_argument('--crop_size', type=int, default=256, help="crop  size")
    parser.add_argument('--im_channel', type=int, default=3, help="input image channel ")
    parser.add_argument('--class_num', type=int, default=1, help="output feature channel")
    parser.add_argument('--epoch', type=int, default=200, help="epochs")
    parser.add_argument('--train_batch', type=int, default=8, help="train_batch")
    parser.add_argument('--val_batch', type=int, default=1, help="val_batch ")
    parser.add_argument('--num_workers', type=int, default=4, help="dataloader numworkers")
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
                        default="bcelog",help="loss name ")
    args = parser.parse_args()
    main(args)
