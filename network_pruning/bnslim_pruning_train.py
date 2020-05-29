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

sys.path.append('../')
from datasets import get_dataloder, datasets_dict
from models import get_models,models_dict
from utils import save_checkpoint, calc_parameters_count, get_logger, get_gpus_memory_info
from utils import BinaryIndicatorsMetric, AverageMeter
from utils import BCEDiceLoss, SoftDiceLoss, DiceLoss


def main(args):
    ############    init config ################
    model_name = args.model
    assert model_name in models_dict.keys(),"The Usage model is not exist !"
    print('Usage model :{}'.format(model_name))

    #################### init logger ###################################
    log_dir = './models/' + args.model+'_'+args.note + '/{}'.format(time.strftime('%Y%m%d-%H%M%S'))
    logger = get_logger(log_dir)
    print('RUNDIR: {}'.format(log_dir))
    logger.info('{}-SlimTrain'.format(args.model))
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
    logger.info("Model Dict has keys: \n {}".format(models_dict.keys()))
    model=get_models(args)
    if torch.cuda.device_count() > 1 and args.use_cuda:
        logger.info('use: %d gpus', torch.cuda.device_count())
        model = nn.DataParallel(model)
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
    else:
        criterion=nn.CrossEntropyLoss()
    if args.use_cuda:
        logger.info("load model and criterion to gpu !")
        model=model.to(args.device)
        criterion=criterion.to(args.device)
    # init optimizer
    #torch.optim.SGD(parametetrs,lr=args.lr,weight_decay=args.weight_decay,momentum=args.momentum)
    optimizer=torch.optim.SGD(model.parameters(),lr=args.lr,weight_decay=args.weight_decay,momentum=args.momentum)
    # init schedulers  Steplr
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epoch)
    # scheduler=torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=30,gamma=0.1,last_epoch=-1)

    #################################### train and val ########################
    max_value=0
    for epoch in range(0,args.epoch):
        # lr=adjust_learning_rate(args,optimizer,epoch)
        scheduler.step()
        logger.info('Epoch: %d lr %e', epoch, scheduler.get_lr()[0])
        # train
        mr, ms, mp, mf, mjc, md, macc, mean_loss=train(args, model, criterion, train_loader,optimizer, epoch, logger)
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
        vmr, vms, vmp, vmf, vmjc, vmd, vmacc, vmean_loss=val(args, model, criterion, val_loader, epoch, logger)

        writer.add_scalar('Val/Loss', vmean_loss, epoch)
        writer.add_scalar('Val/mAcc', vmacc, epoch)
        writer.add_scalar('Val/Recall', vmr, epoch)
        writer.add_scalar('Val/Specifi', vms, epoch)
        writer.add_scalar('Val/Precision', vmp, epoch)
        writer.add_scalar('Val/F1', vmf, epoch)
        writer.add_scalar('Val/Jc', vmjc, epoch)
        writer.add_scalar('Val/Dice', vmd, epoch)

        is_best=True if (vmjc>=max_value) else False
        max_value=max(max_value,vmjc)
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



def updateBN(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(0.0001 * torch.sign(m.weight.data))  # L1 norm



def train(args,model,criterion,train_loader,optimizer,epoch,logger):
    OtherTrain=BinaryIndicatorsMetric()
    model.train()
    train_loss=AverageMeter()
    for step,(input,target,_) in enumerate(train_loader):
        input=input.to(args.device)
        target=target.to(args.device)
        # input is B C H W   target is B,1,H,W  preds: B,1,H,W
        preds=model(input)
        if args.deepsupervision:
            assert isinstance(preds,list) or isinstance(preds,tuple)
            preds=[pred.view(pred.size(0),-1) for pred in preds]
            target = target.view(target.size(0), -1)
            for index in range(len(preds)):
                if index==0:
                    loss=criterion(preds[index],target)
                loss+=criterion(preds[index],target)
            loss/=len(preds)
        else:
            preds=preds.view(preds.size(0),-1)
            target = target.view(target.size(0), -1)
            loss=criterion(preds,target)
        train_loss.update(loss.item(),1)
        optimizer.zero_grad()
        loss.backward()
        updateBN(model)

        optimizer.step()
        # get all the indicators
        if args.deepsupervision:
            OtherTrain.update(labels=target, preds=preds[-1], n=1)
        else:
            OtherTrain.update(labels=target,preds=preds,n=1)
    logger.info("Epoch:{} Train Loss:{:.3f}".format(epoch,train_loss.avg))
    r,s,p,f,jc,dc,acc=OtherTrain.get_current
    logger.info("Acc:{:.3f} Rec:{:.3f} Spe:{:.3f} Pre:{:.3f} F1:{:.3f} Jc:{:.3f} Dice:{:.3f}".format(
        acc,r, s, p, f, jc, dc))
    mr, ms, mp, mf, mjc, md, macc = OtherTrain.get_avg
    return (mr, ms, mp, mf, mjc, md, macc,train_loss.avg)



def val(args,model,criterion,val_loader,epoch,logger):
    OtherVal=BinaryIndicatorsMetric()
    val_loss=AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target,_) in enumerate(val_loader):
            input = input.to(args.device)
            target = target.to(args.device)
            preds =model(input)
            if args.deepsupervision:
                assert isinstance(preds, list) or isinstance(preds, tuple)
                preds = [pred.view(pred.size(0), -1) for pred in preds]
                target = target.view(target.size(0), -1)
                for index in range(len(preds)):
                    if index == 0:
                        loss = criterion(preds[index], target)
                    loss += criterion(preds[index], target)
                loss/=len(preds)
            else:
                preds = preds.view(preds.size(0), -1)
                target = target.view(target.size(0), -1)
                loss = criterion(preds, target)
            val_loss.update(loss.item(),1)
            if args.deepsupervision:
                OtherVal.update(labels=target, preds=preds[-1], n=1)
            else:
                OtherVal.update(labels=target,preds=preds,n=1)

    # update best and recoder early stop
    mr, ms, mp, mf, mjc, md, macc = OtherVal.get_avg
    mean_loss=val_loss.avg
    logger.info("Epoch:{}  Val Loss:{:.3f} ".format(epoch,mean_loss))
    logger.info("Acc:{:.3f} Rec:{:.3f} Spe:{:.3f} Pre:{:.3f} F1:{:.3f} Jc:{:.3f} Dice:{:.3f}".format(macc,mr, ms, mp, mf, mjc, md))
    return mr, ms, mp, mf, mjc, md,macc,mean_loss




if __name__ == '__main__':
    models_name=models_dict.keys()
    datasets_name=datasets_dict.keys()
    parser = argparse.ArgumentParser(description='Unet serieas baseline')
    # Add default argument
    parser.add_argument('--model',  type=str, default='unet',choices=models_name,
                        help='Model to train and evaluation')
    parser.add_argument('--note' ,type=str, default='_',
                        help='model note ')
    parser.add_argument('--dataset',type=str, default='isic2018',choices=datasets_name,
                        help='Model to train and evaluation')
    parser.add_argument('--base_size', type=int, default=256, help="resize base size")
    parser.add_argument('--crop_size', type=int, default=256, help="crop  size")
    parser.add_argument('--im_channel', type=int, default=3, help="input image channel ")
    parser.add_argument('--class_num', type=int, default=1, help="output feature channel")
    parser.add_argument('--epoch', type=int, default=150, help="epochs")
    parser.add_argument('--train_batch', type=int, default=8, help="train_batch")
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
                        default="bcelog",help="loss name ")
    parser.add_argument('--aux_weight', type=float, default=0.2, help=" bce+aux_weight*dice")
    parser.add_argument('--model_optimizer', type=str, choices=['sgd','adm'],default='sgd',
                        help=" model_optimizer ! ")
    parser.add_argument('--lr', type=float, default=1e-3, help=" learning rate we use  ")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help=" learning rate we use  ")
    parser.add_argument('--momentum', type=float, default=0.9, help=" learning rate we use  ")
    parser.add_argument('--schedule', type=int, nargs='+', default=[50, 100],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gammas', type=float, nargs='+', default=[0.5, 0.5],
                        help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
    # for adam
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam
    # resume
    parser.add_argument('--resume',type=str, default=None,help=" resume file path")
    args = parser.parse_args()
    main(args)