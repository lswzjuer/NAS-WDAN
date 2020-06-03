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


import genotypes
from nas_search_unet import BuildNasUnet
from nas_search_unet_prune import BuildNasUnetPrune,BuildNasUnetPrune_l7,BuildNasUnetPrune_full7
#from nas_search_unet_prune_v2 import NasUnetPruneLayer7

sys.path.append('../')
from datasets import get_dataloder,datasets_dict
from utils import save_checkpoint,calc_parameters_count,get_logger,get_gpus_memory_info
from utils import BinaryIndicatorsMetric,AverageMeter
from utils import BCEDiceLoss,SoftDiceLoss,DiceLoss,BCEDiceLoss
from utils import MultiClassEntropyDiceLoss



def main(args):
    #################### init logger ###################################
    log_dir = './eval'+'/{}'.format(args.dataset)+'/{}'.format(args.model)
    logger = get_logger(log_dir)
    print('RUNDIR: {}'.format(log_dir))
    logger.info('{}-Eval'.format(args.model))
    # setting
    args.save_path = log_dir
    args.save_images= os.path.join(args.save_path,"images")
    if not os.path.exists(args.save_images):
        os.mkdir(args.save_images)
    ##################### init device #################################
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    args.use_cuda = args.gpus > 0 and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.use_cuda else 'cpu')
    if args.use_cuda:
        torch.cuda.manual_seed(args.manualSeed)
        cudnn.benchmark = True
    ####################### init dataset ###########################################
    val_loader = get_dataloder(args, split_flag="valid")



    ######################## init model ############################################
    #v_isic
    if args.model == "double_deep_isictrans":
        args.deepsupervision = True
        args.double_down_channel = True
        args.genotype_name = 'stage1_layer9_110epoch_double_deep_final'
        genotype = eval('genotypes.%s' % args.genotype_name)
        model = BuildNasUnetPrune(
            genotype=genotype,
            input_c=args.in_channels,
            c=16,
            num_classes=args.nclass,
            meta_node_num=args.middle_nodes,
            layers=9,
            dp=args.dropout_prob,
            use_sharing=args.use_sharing,
            double_down_channel=args.double_down_channel,
            aux=args.aux
        )
        args.model_path='./logs/chaos/double_deep_isictrans/model_best.pth.tar'
        model.load_state_dict(torch.load(args.model_path,map_location='cpu')['state_dict'])

    # v_mixup0
    elif args.model == "alpha0_double_deep":
        args.deepsupervision = True
        args.double_down_channel = True
        args.genotype_name = 'alpha0_stage1_double_deep_ep200'
        genotype = eval('genotypes.%s' % args.genotype_name)
        model = BuildNasUnetPrune(
            genotype=genotype,
            input_c=args.in_channels,
            c=args.init_channels,
            num_classes=args.nclass,
            meta_node_num=args.middle_nodes,
            layers=args.layers,
            dp=args.dropout_prob,
            use_sharing=args.use_sharing,
            double_down_channel=args.double_down_channel,
            aux=args.aux
        )
        args.model_path = './logs/chaos/alpha0_double_deep/model_best.pth.tar'
        model.load_state_dict(torch.load(args.model_path, map_location='cpu')['state_dict'])
    # v_mixup0.5
    elif args.model == "alpha0_5_double_deep":
        args.deepsupervision = True
        args.double_down_channel = True
        args.genotype_name = 'alpha0_5_stage1_double_deep_ep80'
        genotype = eval('genotypes.%s' % args.genotype_name)
        model = BuildNasUnetPrune(
            genotype=genotype,
            input_c=args.in_channels,
            c=args.init_channels,
            num_classes=args.nclass,
            meta_node_num=args.middle_nodes,
            layers=args.layers,
            dp=args.dropout_prob,
            use_sharing=args.use_sharing,
            double_down_channel=args.double_down_channel,
            aux=args.aux
        )
        args.model_path = './logs/chaos/alpha0_5_double_deep/model_best.pth.tar'
        model.load_state_dict(torch.load(args.model_path, map_location='cpu')['state_dict'])
    # v_mixup1
    elif args.model == "alpha1_double_deep":
        args.deepsupervision = True
        args.double_down_channel = True
        args.genotype_name = 'alpha1_stage1_double_deep_ep200'
        genotype = eval('genotypes.%s' % args.genotype_name)
        model = BuildNasUnetPrune(
            genotype=genotype,
            input_c=args.in_channels,
            c=args.init_channels,
            num_classes=args.nclass,
            meta_node_num=args.middle_nodes,
            layers=args.layers,
            dp=args.dropout_prob,
            use_sharing=args.use_sharing,
            double_down_channel=args.double_down_channel,
            aux=args.aux
        )
        args.model_path = './logs/chaos/alpha1_double_deep/model_best.pth.tar'
        model.load_state_dict(torch.load(args.model_path, map_location='cpu')['state_dict'])

    # v_chaos
    elif args.model == "max_stage1_double_deep":
        args.deepsupervision = True
        args.double_down_channel = True
        args.genotype_name = 'stage0_double_deep_ep80_newim'
        genotype = eval('genotypes.%s' % args.genotype_name)
        model = BuildNasUnetPrune(
            genotype=genotype,
            input_c=args.in_channels,
            c=args.init_channels,
            num_classes=args.nclass,
            meta_node_num=args.middle_nodes,
            layers=args.layers,
            dp=args.dropout_prob,
            use_sharing=args.use_sharing,
            double_down_channel=args.double_down_channel,
            aux=args.aux
        )
        args.model_path = './logs/chaos/stage1_double_deep_newimstage0/model_best.pth.tar'
        model.load_state_dict(torch.load(args.model_path, map_location='cpu')['state_dict'])


    else:
        raise  NotImplementedError()


    setting = {k: v for k, v in args._get_kwargs()}
    logger.info(setting)
    logger.info(genotype)
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
    OtherVal8 = BinaryIndicatorsMetric()
    OtherVal6 = BinaryIndicatorsMetric()
    OtherVal4 = BinaryIndicatorsMetric()
    val_loss = AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (input, target,name) in tqdm(enumerate(val_loader)):
            # input: n,1,h,w   n,1,h,w   name:n list
            input = input.to(args.device)
            target = target.to(args.device)
            preds_list = model(input)

            # save the mask
            file_masks=preds_list[-1].clone()
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
                file_mask.save(os.path.join(path,subdir,file_index+"_8.png"))

            # save the mask
            file_masks=preds_list[-2].clone()
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
                file_mask.save(os.path.join(path,subdir,file_index+"_6.png"))

            # save the mask
            file_masks=preds_list[-3].clone()
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
                file_mask.save(os.path.join(path,subdir,file_index+"_4.png"))


            preds_list = [pred.view(pred.size(0), -1) for pred in preds_list]
            target = target.view(target.size(0), -1)
            v_loss=0
            if args.deepsupervision:
                for pred in preds_list:
                    subloss=criterion(pred,target)
                    v_loss+=subloss
            else:
                v_loss = criterion(preds_list[-1], target)
            val_loss.update(v_loss.item(), 1)
            OtherVal8.update(labels=target, preds=preds_list[-1], n=1)
            OtherVal6.update(labels=target, preds=preds_list[-2], n=1)
            OtherVal4.update(labels=target, preds=preds_list[-3], n=1)
            # batchsize=8



        vmr, vms, vmp, vmf, vmjc, vmd, vmacc = OtherVal8.get_avg
        # mvmr, mvms, mvmp, mvmf, mvmjc, mvmd, mvmacc = valuev2
        logger.info("8:Val_Loss:{:.5f} Acc:{:.5f} Dice:{:.5f} Jc:{:.5f}".format(val_loss.avg, vmacc, vmd, vmjc))
        vmr, vms, vmp, vmf, vmjc, vmd, vmacc = OtherVal6.get_avg
        # mvmr, mvms, mvmp, mvmf, mvmjc, mvmd, mvmacc = valuev2
        logger.info("6:Val_Loss:{:.5f} Acc:{:.5f} Dice:{:.5f} Jc:{:.5f}".format(val_loss.avg, vmacc, vmd, vmjc))
        vmr, vms, vmp, vmf, vmjc, vmd, vmacc = OtherVal4.get_avg
        # mvmr, mvms, mvmp, mvmf, mvmjc, mvmd, mvmacc = valuev2
        logger.info("4:Val_Loss:{:.5f} Acc:{:.5f} Dice:{:.5f} Jc:{:.5f}".format(val_loss.avg, vmacc, vmd, vmjc))




if __name__ == '__main__':
    datasets_name = datasets_dict.keys()
    parser = argparse.ArgumentParser(description='Unet Nas Eval')
    # Add default argument
    parser.add_argument('--model', type=str,
                        help='Model to train and evaluation')
    parser.add_argument('--dataset', type=str, default='chaos', choices=datasets_name,
                        help='Model to train and evaluation')
    parser.add_argument('--note', type=str, default='_', help='train note')
    parser.add_argument('--base_size', type=int, default=256, help="resize base size")
    parser.add_argument('--crop_size', type=int, default=256, help="crop  size")
    parser.add_argument('--in_channels', type=int, default=1, help="input image channel ")
    parser.add_argument('--init_channels', type=int, default=16, help="cell init change channel ")
    parser.add_argument('--nclass', type=int, default=1, help="output feature channel")
    parser.add_argument('--epoch', type=int, default=800, help="epochs")
    parser.add_argument('--val_batch', type=int, default=8, help="val_batch ")
    parser.add_argument('--num_workers', type=int, default=2, help="dataloader numworkers")
    parser.add_argument('--layers', type=int, default=9, help='the layer of the nas search unet')
    parser.add_argument('--middle_nodes', type=int, default=4, help="middle_nodes ")
    parser.add_argument('--dropout_prob', type=int, default=0.0, help="dropout_prob")
    parser.add_argument('--gpus', type=int, default=1, help=" use cuda or not ")
    parser.add_argument('--manualSeed', type=int, default=100, help=" manualSeed ")
    parser.add_argument('--use_sharing', action='store_false', help='normal weight sharing')
    parser.add_argument('--double_down_channel', type=bool, default=True, help=" double_down_channel")
    parser.add_argument('--deepsupervision', type=bool, default=True, help=" like unet++")
    # model special
    parser.add_argument('--aux', action='store_true', help=" deepsupervision of aux layer for  une")
    parser.add_argument('--aux_weight', type=float, default=1, help=" bce+aux_weight*aux_layer_loss")
    parser.add_argument('--genotype_name', type=str, default="FINAL_CELL_GENOTYPE", help="cell genotype")
    parser.add_argument('--loss', type=str, choices=['bce', 'bcelog', 'dice', 'softdice', 'bcedice'],
                        default="bcedice", help="loss name ")
    parser.add_argument('--model_optimizer', type=str, choices=['sgd', 'adm'], default='sgd',
                        help=" model_optimizer ! ")

    args = parser.parse_args()
    main(args)


