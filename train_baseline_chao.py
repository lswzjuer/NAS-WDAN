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
from datasets import get_dataloder,datasets_dict
from models import get_models,models_dict

from utils import save_checkpoint,calc_parameters_count,get_logger,get_gpus_memory_info
from utils import RefugeIndicatorsMetricBinary,BinaryIndicatorsMetric,AverageMeter
from utils import BCEDiceLoss,SoftDiceLoss,DiceLoss,BCEDiceLoss
from utils import MultiClassEntropyDiceLoss


# loss fun and iou/macc/dice fun
def main(args):
    ############    init config ################
    model_name = args.model
    assert model_name in models_dict.keys(),"The Usage model is not exist !"
    print('Usage model :{}'.format(model_name))

    #################### init logger ###################################
    log_dir = './logs' + '/{}/'.format(args.dataset) + args.model+'_'+args.note + '/{}'.format(time.strftime('%Y%m%d-%H%M%S'))
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
        logger.info('Epoch: %d lr %e', epoch, scheduler.get_lr()[0])
        # train
        total_loss=train(args, model, criterion, train_loader,optimizer, epoch, logger)
        # write
        writer.add_scalar('Train/total_loss', total_loss, epoch)

        # val
        tloss, per1,per2=val(args, model, criterion, val_loader, epoch, logger)

        writer.add_scalar('Val/total_loss', tloss, epoch)
        writer.add_scalar('Val/dice', per1[0], epoch)
        writer.add_scalar('Val/iou', per1[1], epoch)
        writer.add_scalar('Val/acc', per1[2], epoch)

        writer.add_scalar('Val_b/dice', per2[0], epoch)
        writer.add_scalar('Val_b/iou', per2[1], epoch)
        writer.add_scalar('Val_b/acc', per2[2], epoch)

        is_best=True if (per1[0]>=max_value) else False
        max_value=max(max_value,per1[0])
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
        # get all the indicators
        # if args.deepsupervision:
        #     OtherTrain.update(preds[-1],target)
        # else:
        #     OtherTrain.update(preds,target)
    logger.info("Epoch:{} TRAIN   T-loss:{:.3f}".format(epoch,tloss_r.avg))
    return tloss_r.avg


def val(args,model,criterion,val_loader,epoch,logger):
    OtherVal_v1=RefugeIndicatorsMetricBinary()
    OtherVal_v2=BinaryIndicatorsMetric()
    tloss_r=AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target,_) in enumerate(val_loader):
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
                tloss = criterion(preds.view(preds.size(0),-1), target.view(target.size(0),-1))
                tloss = tloss * preds.size(0)
            tloss_r.update(tloss.item(), 1)
            if args.deepsupervision:
                OtherVal_v1.update(preds[-1],target)
                OtherVal_v2.update(labels=target.view(target.size(0), -1), preds=preds[-1].view(preds[-1].size(0), -1))
            else:
                OtherVal_v1.update(preds,target)
                OtherVal_v2.update(labels=target.view(target.size(0), -1), preds=preds.view(preds.size(0), -1))
    logger.info(
        "Epoch:{} Val   T-loss:{:.5f} ".format(epoch, tloss_r.avg))

    per1=OtherVal_v1.avg()
    mr, ms, mp, mf, mjc, md, macc=OtherVal_v2.get_avg
    logger.info(" V1     dice:{:.3f} iou:{:.3f} acc:{:.3f}  ".format(per1[0],per1[1],per1[2]))
    logger.info(" V2     dice:{:.3f} iou:{:.3f} acc:{:.3f}  ".format(md,mjc,macc))
    return tloss_r.avg,per1,(md,mjc,macc)





if __name__ == '__main__':
    models_name=models_dict.keys()
    datasets_name=datasets_dict.keys()
    parser = argparse.ArgumentParser(description='Unet serieas baseline')
    # Add default argument
    parser.add_argument('--model',  type=str, default='unet',choices=models_name,
                        help='Model to train and evaluation')
    parser.add_argument('--note' ,type=str, default='_',
                        help='model note ')
    parser.add_argument('--dataset',type=str, default='chaos',choices=datasets_name,
                        help='Model to train and evaluation')
    parser.add_argument('--base_size', type=int, default=256, help="resize base size")
    parser.add_argument('--crop_size', type=int, default=256, help="crop  size")
    parser.add_argument('--im_channel', type=int, default=1, help="input image channel ")
    parser.add_argument('--class_num', type=int, default=1, help="output feature channel")
    parser.add_argument('--epoch', type=int, default=150, help="epochs")
    parser.add_argument('--train_batch', type=int, default=8, help="train_batch")
    parser.add_argument('--val_batch', type=int, default=8, help="val_batch ")
    parser.add_argument('--num_workers', type=int, default=2, help="dataloader numworkers")
    parser.add_argument('--init_weight_type',type=str, choices=["kaiming",'normal','xavier','orthogonal'],
                        default="kaiming",help=" model init mode")
    parser.add_argument('--gpus', type=int,default=1, help=" use cuda or not ")
    parser.add_argument('--grad_clip',type=int, default=5,help=" grid clip to ignore grad boom")
    parser.add_argument('--manualSeed', type=int, default=100, help=" manualSeed ")
    #model special
    parser.add_argument('--deepsupervision', action='store_true', help=" deepsupervision for  unet++")
    parser.add_argument('--time_step',type=int, default=3,help=" r2unet use time step !")
    parser.add_argument('--alpha', type=float, default=1.67, help=" multires unet channel changg ")
    # optimer
    parser.add_argument('--loss',type=str, choices=['bce','bcelog','dice','softdice','bcedice','multibcedice'],
                        default="bcedice",help="loss name ")
    parser.add_argument('--dice_weight',type=float,default=10,help='dice weight')
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