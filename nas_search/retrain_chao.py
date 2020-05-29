import os
import time
import argparse
from tqdm import tqdm
import pickle
import random
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import sys


import genotypes
from nas_search_unet import BuildNasUnet
from nas_search_unet_prune import BuildNasUnetPrune,BuildNasUnetPrune_l7,BuildNasUnetPrune_full7
#from nas_search_unet_prune_v2 import NasUnetPruneLayer7

sys.path.append('../')
from datasets import get_dataloder,datasets_dict
from utils import save_checkpoint,calc_parameters_count,get_logger,get_gpus_memory_info
from utils import RefugeIndicatorsMetricBinary,BinaryIndicatorsMetric,AverageMeter
from utils import BCEDiceLoss,SoftDiceLoss,DiceLoss,BCEDiceLoss
from utils import MultiClassEntropyDiceLoss




# loss fun and iou/macc/dice fun
def main(args):


    #################### init logger ###################################
    log_dir = './logs/' + '{}'.format(args.dataset) + '/{}_{}_{}'.format(args.model,args.note,time.strftime('%Y%m%d-%H%M%S'))

    logger = get_logger(log_dir)
    print('RUNDIR: {}'.format(log_dir))
    logger.info('{}-Train'.format(args.model))
    # setting
    setting={k: v for k, v in args._get_kwargs()}
    logger.info(setting)
    args.save_path = log_dir
    args.save_tbx_log = args.save_path + '/tbx_log'
    writer = SummaryWriter(args.save_tbx_log)
    ##################### init device #################################
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    args.use_cuda= args.gpus>0 and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.use_cuda else 'cpu')
    if args.use_cuda:
        torch.cuda.manual_seed(args.manualSeed)
        cudnn.benchmark = True
    ####################### init dataset ###########################################
    train_loader=get_dataloder(args,split_flag="train")
    val_loader=get_dataloder(args,split_flag="valid")
    ######################## init model ############################################
    # model
    ############init model ###########################
    if  args.model == "nodouble_deep_init32_ep100":
        args.deepsupervision = True
        args.double_down_channel = False
        args.genotype_name = 'nodouble_deep_init32_ep100'
        genotype = eval('genotypes.%s' % args.genotype_name)
        model = BuildNasUnetPrune(
            genotype=genotype,
            input_c=args.in_channels,
            c=32,
            num_classes=args.nclass,
            meta_node_num=args.middle_nodes,
            layers=9,
            dp=args.dropout_prob,
            use_sharing=args.use_sharing,
            double_down_channel=args.double_down_channel,
            aux=args.aux
        )

    elif args.model == "nodouble_deep_isic":
        args.deepsupervision = True
        args.double_down_channel = False
        args.genotype_name = 'stage1_layer9_110epoch_deep_final'
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

    elif args.model == "nodouble_deep_drop02_layer7end":
        args.deepsupervision = True
        args.double_down_channel = False
        args.genotype_name = 'nodouble_deep_drop02_layer7end'
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

    elif args.model == "stage1_nodouble_deep_ep36":
        args.deepsupervision = True
        args.double_down_channel = False
        args.genotype_name = 'stage1_nodouble_deep_ep36'
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

    elif args.model == "stage1_nodouble_deep_ep63":
        args.deepsupervision = True
        args.double_down_channel = False
        args.genotype_name = 'stage1_nodouble_deep_ep63'
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
    elif args.model == "stage1_nodouble_deep_ep83":
        args.deepsupervision = True
        args.double_down_channel = False
        args.genotype_name = 'stage1_nodouble_deep_ep83'
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



    elif args.model == "alpha1_stage1_double_deep_ep80":
        args.deepsupervision = True
        args.double_down_channel = True
        args.genotype_name = 'alpha1_stage1_double_deep_ep80'
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

    elif args.model == "alpha0_stage1_double_deep_ep80":
        args.deepsupervision = True
        args.double_down_channel = True
        args.genotype_name = 'alpha0_stage1_double_deep_ep80'
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

    elif args.model == "alpha0_5_stage1_double_deep_ep80":
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



    # isic trans
    elif args.model == "stage1_layer9_110epoch_double_deep_final":
        args.deepsupervision = True
        args.double_down_channel = True
        args.genotype_name = 'stage1_layer9_110epoch_double_deep_final'
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

    #chaos
    elif args.model == "stage0_double_deep_ep80_newim":
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

    elif args.model == "stage1_double_deep_ep80":
        args.deepsupervision = True
        args.double_down_channel = True
        args.genotype_name = 'stage1_double_deep_ep80'
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


    elif args.model == "stage1_double_deep_ep80_ts":
        args.deepsupervision = True
        args.double_down_channel = True
        args.genotype_name = 'stage1_double_deep_ep80_ts'
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

    # cvc trans
    elif args.model == "layer7_double_deep":
        args.deepsupervision = True
        args.double_down_channel = True
        args.genotype_name = 'layer7_double_deep'
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

    if torch.cuda.device_count() > 1 and args.use_cuda:
        logger.info('use: %d gpus', torch.cuda.device_count())
        model = nn.DataParallel(model)
    setting = {k: v for k, v in args._get_kwargs()}
    logger.info(setting)
    logger.info(genotype)
    logger.info('param size = %fMB', calc_parameters_count(model))


    # init loss
    if args.loss=='bce':
        criterion=nn.BCELoss()
    elif args.loss=='bcelog':
        criterion=nn.BCEWithLogitsLoss()
    elif args.loss=="dice":
        criterion=DiceLoss()
    elif args.loss=="softdice":
        criterion=SoftDiceLoss()
    elif args.loss=='bcedice':
        criterion=BCEDiceLoss()
    elif args.loss=='multibcedice':
        criterion=MultiClassEntropyDiceLoss()
    else:
        criterion=nn.CrossEntropyLoss()
    if args.use_cuda:
        logger.info("load model and criterion to gpu !")
        model=model.to(args.device)
        criterion=criterion.to(args.device)
    # init optimizer
    optimizer=torch.optim.SGD(model.parameters(),lr=args.lr,weight_decay=args.weight_decay,momentum=args.momentum)
    # init schedulers  Steplr
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epoch)
    # scheduler=torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=30,gamma=0.1,last_epoch=-1)
    ############################### check resume #########################
    start_epoch=0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            logger.info("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=args.device)
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            raise FileNotFoundError("No checkpoint found at '{}'".format(args.resume))

    #################################### train and val ########################


    max_value=0
    for epoch in range(start_epoch,args.epoch):
        # lr=adjust_learning_rate(args,optimizer,epoch)
        scheduler.step()
        # logger.info('Epoch: %d lr %e', epoch, scheduler.get_lr()[0])
        # train
        total_loss=train(args, model, criterion, train_loader,optimizer, epoch, logger)
        # write
        writer.add_scalar('Train/total_loss', total_loss, epoch)
        # val
        tloss, md=val(args, model, criterion, val_loader, epoch, logger)
        writer.add_scalar('Val/total_loss', tloss, epoch)

        is_best=True if (md>=max_value) else False
        max_value=max(max_value,md)
        state={
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'scheduler': model.state_dict(),
            }
        logger.info("epoch:{} best:{} max_value:{}".format(epoch,is_best,max_value))
        if not is_best:
            torch.save(state,os.path.join(args.save_path,"checkpoint.pth.tar"))
        else:
            torch.save(state,os.path.join(args.save_path,"checkpoint.pth.tar"))
            torch.save(state,os.path.join(args.save_path,"model_best.pth.tar"))
    writer.close()


# compute func change
def train(args,model,criterion,train_loader,optimizer,epoch,logger):
    model.train()
    tloss_r=AverageMeter()
    for step,(input,target,_) in enumerate(train_loader):
        # n c h w,  n h w
        input=input.to(args.device)
        target=target.to(args.device)
        preds=model(input)
        if args.deepsupervision:
            assert isinstance(preds,list) or isinstance(preds,tuple)
            tloss=0
            for index in range(len(preds)):
                subtloss=criterion(preds[index].view(preds[index].size(0),-1), target.view(target.size(0),-1))
                tloss+=subtloss
            tloss=tloss*preds[-1].size(0)
            tloss/=len(preds)
        else:
            tloss=criterion(preds.view(preds.size(0),-1), target.view(target.size(0),-1))
            tloss=tloss*preds.size(0)
        tloss_r.update(tloss.item(),1)
        optimizer.zero_grad()
        tloss.backward()
        optimizer.step()
    logger.info("Epoch:{} TRAIN   T-loss:{:.3f}".format(epoch,tloss_r.avg))
    return tloss_r.avg





def val(args,model,criterion,val_loader,epoch,logger):
    OtherVal=BinaryIndicatorsMetric()
    tloss_r=AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target,_) in enumerate(val_loader):
            # n,1,h,w  n,h,w
            input = input.to(args.device)
            target = target.to(args.device)
            preds = model(input)
            if args.deepsupervision:
                assert isinstance(preds, list) or isinstance(preds, tuple)
                tloss = 0
                for index in range(len(preds)):
                    subtloss = criterion(preds[index].view(preds[index].size(0),-1), target.view(target.size(0),-1))
                    tloss += subtloss
                tloss = tloss * preds[-1].size(0)
                tloss /= len(preds)
            else:
                tloss = criterion(preds[-1].view(preds[-1].size(0),-1), target.view(target.size(0),-1))
                tloss = tloss * preds.size(0)
            tloss_r.update(tloss.item(), 1)
            OtherVal.update(labels=target.view(target.size(0),-1),preds=preds[-1].view(preds[-1].size(0),-1))

    logger.info( "Epoch:{} Val   T-loss:{:.5f} ".format(epoch, tloss_r.avg))
    mr, ms, mp, mf, mjc, md, macc=OtherVal.get_avg
    logger.info(" Val     dice:{:.3f} iou:{:.3f} acc:{:.3f}  ".format(md,mjc,macc))
    return tloss_r.avg,md






if __name__ == '__main__':
    datasets_name = datasets_dict.keys()
    parser = argparse.ArgumentParser(description='Unet Nas baseline')
    # Add default argument
    parser.add_argument('--model', type=str,default='stage1_nodouble_deep_ep36',help='Model to train and evaluation')
    parser.add_argument('--dataset', type=str, default='chaos', choices=datasets_name,
                        help='Model to train and evaluation')
    parser.add_argument('--note', type=str, default='_', help='train note')
    parser.add_argument('--base_size', type=int, default=256, help="resize base size")
    parser.add_argument('--crop_size', type=int, default=256, help="crop  size")
    parser.add_argument('--in_channels', type=int, default=1, help="input image channel ")
    parser.add_argument('--init_channels', type=int, default=16, help="cell init change channel ")
    parser.add_argument('--nclass', type=int, default=1, help="output feature channel")
    parser.add_argument('--epoch', type=int, default=300, help="epochs")
    parser.add_argument('--train_batch', type=int, default=8, help="train_batch")
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
    parser.add_argument('--loss', type=str, choices=['bce', 'bcelog', 'dice', 'softdice', 'bcedice','multibcedice'],
                        default="bcedice", help="loss name ")
    parser.add_argument('--model_optimizer', type=str, choices=['sgd', 'adm'], default='sgd',
                        help=" model_optimizer ! ")

    parser.add_argument('--lr', type=float, default=2e-3, help=" learning rate we use  ")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help=" learning rate we use  ")
    parser.add_argument('--momentum', type=float, default=0.9, help=" learning rate we use  ")
    parser.add_argument('--schedule', type=int, nargs='+', default=[50, 100],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gammas', type=float, nargs='+', default=[0.5, 0.5],
                        help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
    # for adam
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    # resume
    parser.add_argument('--resume', type=str, default=None, help=" resume file path")
    args = parser.parse_args()
    main(args)