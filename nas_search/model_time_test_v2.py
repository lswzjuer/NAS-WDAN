import os
import time
import argparse
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
import tqdm

import genotypes
from prune_dd import  BuildNasUnetPrune as net_dd
from prune_double import BuildNasUnetPrune as net_double
from prune_nodouble import BuildNasUnetPrune as net_nodouble
from prune_nodouble_deep import BuildNasUnetPrune as net_nodouble_deep
from nas_search_unet_prune import  BuildNasUnetPrune



sys.path.append('../')
from datasets import get_dataloder
from utils import calc_parameters_count



def test_time(args,model,dataloader):
    count = 0
    model.eval()
    model=model.to(args.device)
    start = time.time()
    for step, (input, target,_) in enumerate(dataloader):
        input = input.to(args.device)
        target = target.to(args.device)
        output = model(input)
        count += args.train_batch
        if step > 30:
            break
    end_tim = time.time()
    return (end_tim-start)/count


def main(args):

    args.device = torch.device('cuda')
    args.dataset = 'isic2018'
    args.train_batch = 2
    args.val_batch = 2
    args.num_workers = 2
    args.crop_size = 256
    args.base_size = 256
    train_loader = get_dataloder(args, split_flag="train")
    #args.model = 'nodouble_deep'
    ######################## init model ############################################
    # model
    # get the network parameters
    if args.model == 'dd':
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
        print('param size = %fMB', calc_parameters_count(model))

    elif args.model == 'double':
        args.deepsupervision = False
        args.double_down_channel = True
        args.genotype_name = 'stage1_layer9_110epoch_double_final'
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
        print('param size = %fMB', calc_parameters_count(model))

    elif args.model == 'nodouble':
        args.deepsupervision = False
        args.double_down_channel = False
        args.genotype_name = 'stage1_layer9_110epoch_final'
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
        print('param size = %fMB', calc_parameters_count(model))

    elif args.model == 'nodouble_deep':
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
        print('param size = %fMB', calc_parameters_count(model))

    else:
        raise  NotImplementedError()
    time=test_time(args,model,train_loader)
    print("Infrence time:{}".format(time))




if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Unet Nas Prune')
    # Add default argument
    parser.add_argument('--model', type=str, default='dd', choices=['dd', 'double', 'nodouble', 'nodouble_deep'],
                        help='Model to train and evaluation')
    parser.add_argument('--dataset', type=str, default='isic2018', help='Model to train and evaluation')
    parser.add_argument('--note', type=str, default='_', help='train note')
    parser.add_argument('--base_size', type=int, default=256, help="resize base size")
    parser.add_argument('--crop_size', type=int, default=256, help="crop  size")
    parser.add_argument('--in_channels', type=int, default=3, help="input image channel ")
    parser.add_argument('--init_channels', type=int, default=16, help="cell init change channel ")
    parser.add_argument('--nclass', type=int, default=1, help="output feature channel")
    parser.add_argument('--epoch', type=int, default=300, help="epochs")
    parser.add_argument('--train_batch', type=int, default=1, help="train_batch")
    parser.add_argument('--val_batch', type=int, default=1, help="val_batch ")
    parser.add_argument('--num_workers', type=int, default=2, help="dataloader numworkers")
    parser.add_argument('--layers', type=int, default=9, help='the layer of the nas search unet')
    parser.add_argument('--middle_nodes', type=int, default=4, help="middle_nodes ")
    parser.add_argument('--dropout_prob', type=int, default=0.0, help="dropout_prob")
    parser.add_argument('--print_freq', type=int, default=50, help=" print freq (iteras) ")
    parser.add_argument('--gpus', type=int, default=1, help=" use cuda or not ")
    parser.add_argument('--manualSeed', type=int, default=100, help=" manualSeed ")
    parser.add_argument('--use_sharing', action='store_false', help='normal weight sharing')
    parser.add_argument('--double_down_channel', type=bool, default=True, help=" double_down_channel")
    parser.add_argument('--deepsupervision', type=bool, default=True, help=" like unet++")
    # model special
    parser.add_argument('--aux', action='store_true', help=" deepsupervision of aux layer for  une")
    parser.add_argument('--aux_weight', type=float, default=1, help=" bce+aux_weight*aux_layer_loss")
    parser.add_argument('--genotype_name', type=str, default="stage1_layer9_110epoch_double_deep_final",
                        help="cell genotype")
    parser.add_argument('--loss', type=str, choices=['bce', 'bcelog', 'dice', 'softdice', 'bcedice'],
                        default="bcelog", help="loss name ")
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