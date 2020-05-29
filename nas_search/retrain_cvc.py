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

import genotypes
from nas_search_unet import BuildNasUnet
from nas_search_unet_prune import BuildNasUnetPrune
# from nas_search_unet_prune_v2 import NasUnetPruneLayer7
from cvc_slimv3 import  BuildNasUnetPrune as   BuildNasUnetPruneSlim

from feature_branch_test import BuildNasUnetPruneNormal,BuildNasUnetPruneNormalDown,BuildNasUnetPruneNormalUp

sys.path.append('../')
from datasets import get_dataloder, datasets_dict
from utils import save_checkpoint, calc_parameters_count, get_logger, get_gpus_memory_info
from utils import BinaryIndicatorsMetric, AverageMeter
from utils import BCEDiceLoss, SoftDiceLoss, DiceLoss


def main(args):
    #################### init logger ###################################
    log_dir = './logs/' + '{}'.format(args.dataset) + '/{}_{}_{}'.format(args.model,args.note,time.strftime('%Y%m%d-%H%M%S'))


    logger = get_logger(log_dir)
    print('RUNDIR: {}'.format(log_dir))
    logger.info('{}-Train'.format(args.model))
    # setting
    args.save_path = log_dir
    args.save_tbx_log = args.save_path + '/tbx_log'
    writer = SummaryWriter(args.save_tbx_log)
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
    train_loader = get_dataloder(args, split_flag="train")
    val_loader = get_dataloder(args, split_flag="valid")

    ############init model ###########################
    if  args.model == "layer7_double_deep":
        args.deepsupervision = True
        args.double_down_channel = True
        args.genotype_name = 'layer7_double_deep'
        model_alphas = None
        genotype = eval('genotypes.%s' % args.genotype_name)
        model = BuildNasUnetPrune(
            genotype=genotype,
            input_c=args.in_channels,
            c=args.init_channels,
            num_classes=args.nclass,
            meta_node_num=args.middle_nodes,
            layers=7,
            dp=args.dropout_prob,
            use_sharing=args.use_sharing,
            double_down_channel=args.double_down_channel,
            aux=args.aux
        )



    elif args.model == "stage1_double_deep":
        args.deepsupervision = True
        args.double_down_channel = True
        args.genotype_name = 'stage1_double_deep'
        model_alphas = None
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

    elif args.model == "stage1_nodouble_deep":
        args.deepsupervision = True
        args.double_down_channel = False
        args.genotype_name = 'stage1_deep'
        model_alphas = None
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

    elif args.model == "stage1_nodouble_deep_slim":
        args.deepsupervision = True
        args.double_down_channel = False
        args.genotype_name = 'stage1_deep'
        model_alphas = None
        genotype = eval('genotypes.%s' % args.genotype_name)
        model = BuildNasUnetPruneSlim(
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
        model_alphas = None
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
        model_alphas = None
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

    #isic trans
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


    # just normaL cell keep
    elif args.model == "dd_normal":
        args.deepsupervision = True
        args.double_down_channel = True
        args.genotype_name = 'alpha0_5_stage1_double_deep_ep80'
        genotype = eval('genotypes.%s' % args.genotype_name)
        model = BuildNasUnetPruneNormal(
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

    # normal+down
    elif args.model == "dd_normaldown":
        args.deepsupervision = True
        args.double_down_channel = True
        args.genotype_name = 'alpha0_5_stage1_double_deep_ep80'
        genotype = eval('genotypes.%s' % args.genotype_name)
        model = BuildNasUnetPruneNormalDown(
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

    # normal+up 
    elif args.model == "dd_normalup":
        args.deepsupervision = True
        args.double_down_channel = True
        args.genotype_name = 'alpha0_5_stage1_double_deep_ep80'
        genotype = eval('genotypes.%s' % args.genotype_name)
        model = BuildNasUnetPruneNormalUp(
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

    # normal+up+down
    elif args.model == "alpha0_5_stage1_double_deep_ep80":
        args.deepsupervision = True
        args.double_down_channel = True
        args.genotype_name = 'alpha0_5_stage1_double_deep_ep80'
        model_alphas = None
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

    # abliation study of channel doubling and deepsupervision
    elif args.model == "alpha0_5_stage1_double_nodeep_ep80":
        args.deepsupervision = False
        args.double_down_channel = True
        args.genotype_name = 'alpha0_5_stage1_double_nodeep_ep80'
        model_alphas = None
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

    elif args.model == "alpha0_5_stage1_nodouble_deep_ep80":
        args.deepsupervision = True
        args.double_down_channel = False
        args.genotype_name = 'alpha0_5_stage1_nodouble_deep_ep80'
        model_alphas = None
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

    elif args.model == "alpha0_5_stage1_nodouble_nodeep_ep80":
        args.deepsupervision = False
        args.double_down_channel = False
        args.genotype_name = 'alpha0_5_stage1_nodouble_nodeep_ep80'
        model_alphas = None
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
    # init optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    # init schedulers  Steplr
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
    # scheduler=torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=30,gamma=0.1,last_epoch=-1)

    ############################### check resume #########################
    start_epoch = 0
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
    max_value = 0
    for epoch in range(start_epoch, args.epoch):
        # lr=adjust_learning_rate(args,optimizer,epoch)
        scheduler.step()
        logger.info('Epoch: %d lr %e', epoch, scheduler.get_lr()[0])
        # train
        if args.deepsupervision:
            mean_loss, value1, value2 = train(args, model, criterion, train_loader, optimizer)
            mr, ms, mp, mf, mjc, md, macc = value1
            mmr, mms, mmp, mmf, mmjc, mmd, mmacc = value2
            logger.info(
                "Epoch:{} Train_Loss:{:.3f} Acc:{:.3f} Dice:{:.3f} Jc:{:.3f}".format(epoch, mean_loss, macc, md, mjc))
            logger.info("                        dmAcc:{:.3f} dmDice:{:.3f} dmJc:{:.3f}".format(mmacc, mmd, mmjc))
            writer.add_scalar('Train/dmAcc', mmacc, epoch)
            writer.add_scalar('Train/dRecall', mmr, epoch)
            writer.add_scalar('Train/dSpecifi', mms, epoch)
            writer.add_scalar('Train/dPrecision', mmp, epoch)
            writer.add_scalar('Train/dF1', mmf, epoch)
            writer.add_scalar('Train/dJc', mmjc, epoch)
            writer.add_scalar('Train/dDice', mmd, epoch)
        else:
            mean_loss, value1 = train(args, model, criterion, train_loader,
                                      optimizer)
            mr, ms, mp, mf, mjc, md, macc = value1
            logger.info(
                "Epoch:{} Train_Loss:{:.3f} Acc:{:.3f} Dice:{:.3f} Jc:{:.3f}".format(epoch, mean_loss, macc, md, mjc))
        # write
        writer.add_scalar('Train/Loss', mean_loss, epoch)
        writer.add_scalar('Train/mAcc', macc, epoch)
        writer.add_scalar('Train/Recall', mr, epoch)
        writer.add_scalar('Train/Specifi', ms, epoch)
        writer.add_scalar('Train/Precision', mp, epoch)
        writer.add_scalar('Train/F1', mf, epoch)
        writer.add_scalar('Train/Jc', mjc, epoch)
        writer.add_scalar('Train/Dice', md, epoch)

        # val
        if args.deepsupervision:
            vmean_loss, valuev1, valuev2 = infer(args, model, criterion, val_loader)
            vmr, vms, vmp, vmf, vmjc, vmd, vmacc = valuev1
            mvmr, mvms, mvmp, mvmf, mvmjc, mvmd, mvmacc = valuev2
            logger.info(
                "Epoch:{} Val_Loss:{:.3f} Acc:{:.3f} Dice:{:.3f} Jc:{:.3f}".format(epoch, vmean_loss, vmacc, vmd, vmjc))
            logger.info("                        dmAcc:{:.3f} dmDice:{:.3f} dmJc:{:.3f}".format(mvmacc, mvmd, mvmjc))
            writer.add_scalar('Val/mAcc', mvmacc, epoch)
            writer.add_scalar('Val/Recall', mvmr, epoch)
            writer.add_scalar('Val/Specifi', mvms, epoch)
            writer.add_scalar('Val/Precision', mvmp, epoch)
            writer.add_scalar('Val/F1', mvmf, epoch)
            writer.add_scalar('Val/Jc', mvmjc, epoch)
            writer.add_scalar('Val/Dice', mvmd, epoch)
        else:
            vmean_loss, valuev1 = infer(args, model, criterion, val_loader)
            vmr, vms, vmp, vmf, vmjc, vmd, vmacc = valuev1
            logger.info(
                "Epoch:{} Val_Loss:{:.3f} Acc:{:.3f} Dice:{:.3f} Jc:{:.3f}".format(epoch, vmean_loss, vmacc, vmd, vmjc))

        is_best = True if (vmjc >=max_value) else False
        max_value = max(max_value, vmjc)
        writer.add_scalar('Val/Loss', vmean_loss, epoch)
        writer.add_scalar('Val/mAcc', vmacc, epoch)
        writer.add_scalar('Val/Recall', vmr, epoch)
        writer.add_scalar('Val/Specifi', vms, epoch)
        writer.add_scalar('Val/Precision', vmp, epoch)
        writer.add_scalar('Val/F1', vmf, epoch)
        writer.add_scalar('Val/Jc', vmjc, epoch)
        writer.add_scalar('Val/Dice', vmd, epoch)

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


def train(args, model, criterion, train_loader, optimizer):
    Train_recoder = BinaryIndicatorsMetric()
    Deep_Train_recoder = BinaryIndicatorsMetric()
    loss_recoder = AverageMeter()
    model.train()
    for step, (input, target,_) in tqdm(enumerate(train_loader)):
        input = input.to(args.device)
        target = target.to(args.device)
        # input is B C H W   target is B,1,H,W  preds: B,1,H,W
        optimizer.zero_grad()
        # [output1,...]
        preds_list = model(input)
        preds_list = [pred.view(pred.size(0), -1) for pred in preds_list]
        target = target.view(target.size(0), -1)
        if args.deepsupervision:
            for i in range(len(preds_list)):
                if i == 0:
                    w_loss = criterion(preds_list[i], target)
                w_loss += criterion(preds_list[i], target)
        else:
            w_loss = criterion(preds_list[-1], target)
        w_loss.backward()
        optimizer.step()
        loss_recoder.update(w_loss.item(), 1)
        # get all the indicators
        # Deep_Train_recoder
        if args.deepsupervision:
            avg_pred = 0
            for pred in preds_list:
                avg_pred += pred
            avg_pred /= len(preds_list)
            Deep_Train_recoder.update(labels=target, preds=avg_pred, n=1)
        Train_recoder.update(labels=target, preds=preds_list[-1], n=1)
    if args.deepsupervision:
        return loss_recoder.avg, Train_recoder.get_avg, Deep_Train_recoder.get_avg
    else:
        return loss_recoder.avg, Train_recoder.get_avg


def infer(args, model, criterion, val_loader):
    OtherVal = BinaryIndicatorsMetric()
    DeepOtherVal = BinaryIndicatorsMetric()
    val_loss = AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target,_) in tqdm(enumerate(val_loader)):
            input = input.to(args.device)
            target = target.to(args.device)
            preds_list = model(input)
            preds_list = [pred.view(pred.size(0), -1) for pred in preds_list]
            target = target.view(target.size(0), -1)
            if args.deepsupervision:
                for i in range(len(preds_list)):
                    if i == 0:
                        v_loss = criterion(preds_list[i], target)
                    v_loss += criterion(preds_list[i], target)
            else:
                v_loss = criterion(preds_list[-1], target)
            val_loss.update(v_loss.item(), 1)
            # get the deepsupervision
            if args.deepsupervision:
                avg_pred = 0
                for pred in preds_list:
                    avg_pred += pred
                avg_pred /= len(preds_list)
                DeepOtherVal.update(labels=target, preds=avg_pred, n=1)
            OtherVal.update(labels=target, preds=preds_list[-1], n=1)
        if args.deepsupervision:
            return val_loss.avg, OtherVal.get_avg, DeepOtherVal.get_avg
        else:
            return val_loss.avg, OtherVal.get_avg





if __name__ == '__main__':
    datasets_name = datasets_dict.keys()
    parser = argparse.ArgumentParser(description='Unet Nas baseline')
    # Add default argument
    parser.add_argument('--model', type=str,
                        choices=['layer7_double_deep', 'layer9_double_deep', 'stage1_double_deep', 'stage1_nodouble_deep',
                                 ], default='layer7_double_deep',
                        help='Model to train and evaluation')
    parser.add_argument('--dataset', type=str, default='cvc', choices=datasets_name,
                        help='Model to train and evaluation')
    parser.add_argument('--note', type=str, default='_', help='train note')
    parser.add_argument('--base_size', type=int, default=256, help="resize base size")
    parser.add_argument('--crop_size', type=int, default=256, help="crop  size")
    parser.add_argument('--in_channels', type=int, default=3, help="input image channel ")
    parser.add_argument('--init_channels', type=int, default=16, help="cell init change channel ")
    parser.add_argument('--nclass', type=int, default=1, help="output feature channel")
    parser.add_argument('--epoch', type=int, default=1600, help="epochs")
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
    parser.add_argument('--loss', type=str, choices=['bce', 'bcelog', 'dice', 'softdice', 'bcedice'],
                        default="bcedice", help="loss name ")
    parser.add_argument('--model_optimizer', type=str, choices=['sgd', 'adm'], default='sgd',
                        help=" model_optimizer ! ")

    parser.add_argument('--lr', type=float, default=5e-3, help=" learning rate we use  ")
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


