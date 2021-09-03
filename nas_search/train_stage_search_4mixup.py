import os
import time
import argparse
from tqdm import tqdm
import pickle
import copy
import sys
import random
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import torch.utils.data as data
import torch.nn.functional as F

from genotypes import CellLinkDownPos, CellLinkUpPos, CellPos
from nas_model import get_models

sys.path.append('../')
from datasets import get_dataloder, datasets_dict
from utils import save_checkpoint, calc_parameters_count, get_logger, get_gpus_memory_info
from utils import BinaryIndicatorsMetric, AverageMeter
from utils import BCEDiceLoss, SoftDiceLoss, DiceLoss



def main(args):
    ############    init config ################
    #################### init logger ###################################
    log_dir = './search_exp/' + '/{}'.format(args.model) + \
              '/{}'.format(args.dataset) + '/{}_{}'.format(time.strftime('%Y%m%d-%H%M%S'),args.note)

    logger = get_logger(log_dir)
    print('RUNDIR: {}'.format(log_dir))
    logger.info('{}-Search'.format(args.model))
    args.save_path = log_dir
    args.save_tbx_log = args.save_path + '/tbx_log'
    writer = SummaryWriter(args.save_tbx_log)
    ##################### init device #################################
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    args.use_cuda = args.gpus > 0 and torch.cuda.is_available()
    args.multi_gpu = args.gpus > 1 and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.use_cuda else 'cpu')
    if args.use_cuda:
        torch.cuda.manual_seed(args.manualSeed)
        cudnn.enabled = True
        cudnn.benchmark = True
    setting = {k: v for k, v in args._get_kwargs()}
    logger.info(setting)

    ####################### init dataset ###########################################
    assert args.dataset=="total"
    logger.info("Dataset for search is {}".format(args.dataset))
    train_dataset = datasets_dict[args.dataset](args, split='train')
    val_dataset = datasets_dict[args.dataset](args, split='valid')
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))
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
        logger.info("load criterion to gpu !")
    criterion = criterion.to(args.device)
    ######################## init model ############################################
    switches_normal = []
    switches_down = []
    switches_up = []
    nums_mixop=sum([2+i for i in range(args.meta_node_num)])
    for i in range(nums_mixop):
        switches_normal.append([True for j in range(len(CellPos))])
    for i in range(nums_mixop):
        switches_down.append([True for j in range(len(CellLinkDownPos))])
    for i in range(nums_mixop):
        switches_up.append([True for j in range(len(CellLinkUpPos))])
    # 6-->3-->1
    drop_op_down=[2,3]
    # 4-->2-->1
    drop_op_up=[2,1]
    # 7-->4-->1
    drop_op_normal=[3,3]
    # stage0 pruning  stage 1 pruning, stage 2 (training)
    original_train_batch=args.train_batch
    original_val_batch=args.val_batch
    for sp in range(2):
        # build dataloader
        # model ,numclass=1,im_ch=3,init_channel=16,intermediate_nodes=4,layers=9
        if sp==0:
            args.model="UnetLayer7"
            args.layers=7
            sp_train_batch=original_train_batch
            sp_val_batch=original_val_batch
            sp_epoch=args.epochs
            sp_lr=args.lr
        else:
            args.model = "UnetLayer9"
            args.layers=9
            sp_train_batch = original_train_batch
            sp_val_batch = original_val_batch
            sp_lr = args.lr
            sp_epoch = args.epochs

        train_queue = data.DataLoader(train_dataset,
                                      batch_size=sp_train_batch,
                                      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                                      pin_memory=True,
                                      num_workers=args.num_workers
                                      )
        val_queue = data.DataLoader(train_dataset,
                                    batch_size=sp_train_batch,
                                    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
                                    pin_memory=True,
                                    num_workers=args.num_workers
                                    )
        test_dataloader = data.DataLoader(val_dataset,
                                          batch_size=sp_val_batch,
                                          pin_memory=True,
                                          num_workers=args.num_workers
                                          )
        logger.info("stage:{} model:{} epoch:{} lr:{} train_batch:{} val_batch:{}".format(sp,
                                            args.model,sp_epoch,sp_lr,sp_train_batch,sp_val_batch))

        model = get_models(args, switches_normal, switches_down, switches_up)
        save_model_path=os.path.join(args.save_path,"stage_{}_model".format(sp))
        if not os.path.exists(save_model_path):
            os.mkdir(save_model_path)
        if args.multi_gpu:
            logger.info('use: %d gpus', args.gpus)
            model = nn.DataParallel(model)
        model = model.to(args.device)
        logger.info('param size = %fMB', calc_parameters_count(model))
        # init optimizer for arch parameters and weight parameters
        # final stage, just train the network parameters
        optimizer_arch = torch.optim.Adam(model.module.arch_parameters(), lr=args.arch_lr, betas=(0.5, 0.999),
                                            weight_decay=args.arch_weight_decay)
        optimizer_weight = torch.optim.SGD(model.module.weight_parameters(), lr=sp_lr, weight_decay=args.weight_decay,
                                           momentum=args.momentum)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_weight, sp_epoch, eta_min=args.lr_min)
        #################################### train and val ########################
        max_value=0
        sp_recoder_list=[]
        for epoch in range(0, sp_epoch):
            # lr=adjust_learning_rate(args,optimizer,epoch)
            scheduler.step()
            logger.info('Epoch: %d lr %e', epoch, scheduler.get_lr()[0])
            # train
            if epoch < args.arch_after:
                weight_loss_avg, arch_loss_avg, mr, ms, mp, mf, mjc, md, macc = train(args, train_queue, val_queue,
                                                                                      model, criterion,
                                                                                      optimizer_weight, optimizer_arch,
                                                                                      train_arch=False)
            else:
                weight_loss_avg, arch_loss_avg, mr, ms, mp, mf, mjc, md, macc = train(args, train_queue, val_queue,
                                                                                      model, criterion,
                                                                                      optimizer_weight, optimizer_arch,
                                                                                      train_arch=True)
            logger.info("Epoch:{} WeightLoss:{:.3f}  ArchLoss:{:.3f}".format(epoch, weight_loss_avg, arch_loss_avg))
            logger.info("         Acc:{:.3f}   Dice:{:.3f}  Jc:{:.3f}".format(macc, md, mjc))
            # write
            writer.add_scalar('stage{}_Train/W_loss'.format(sp), weight_loss_avg, epoch)
            writer.add_scalar('staget{}_Train/A_loss'.format(sp), arch_loss_avg, epoch)
            writer.add_scalar('stage{}_Train/mAcc'.format(sp), macc, epoch)
            writer.add_scalar('stage{}_Train/Recall'.format(sp), mr, epoch)
            writer.add_scalar('stage{}_Train/Specifi'.format(sp), ms, epoch)
            writer.add_scalar('stage{}_Train/Precision'.format(sp), mp, epoch)
            writer.add_scalar('stage{}_Train/F1'.format(sp), mf, epoch)
            writer.add_scalar('stage{}_Train/Jc'.format(sp), mjc, epoch)
            writer.add_scalar('stage{}_Train/Dice'.format(sp), md, epoch)
            # infer
            if (epoch + 1) % args.infer_epoch == 0:
                genotype = model.module.genotype()
                logger.info('genotype = %s', genotype)
                val_loss, (vmr, vms, vmp, vmf, vmjc, vmd, vmacc) = infer(args, model, val_queue, criterion)
                logger.info("ValLoss:{:.3f} ValAcc:{:.3f}  ValDice:{:.3f} ValJc:{:.3f}".format(val_loss, vmacc, vmd, vmjc))
                writer.add_scalar('stage{}_Val/loss'.format(sp), val_loss, epoch)
                writer.add_scalar('stage{}_Val/mAcc'.format(sp), vmacc, epoch)
                writer.add_scalar('stage{}_Val/mDice'.format(sp), vmd, epoch)
                writer.add_scalar('stage{}_Val/mDice'.format(sp), vmjc, epoch)

                sp_recoder_list.append((epoch,genotype,vmd,vmjc))

                is_best=True if (vmjc >=max_value) else False
                max_value=max(max_value,vmjc)
                state = {
                    'epoch': epoch,
                    'optimizer_arch': optimizer_arch.state_dict(),
                    'optimizer_weight': optimizer_weight.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'state_dict': model.module.state_dict(),
                    'alphas_dict': model.module.alphas_dict(),
                }
                logger.info("epoch:{} best:{} max_value:{}".format(epoch, is_best, max_value))
                if not is_best:
                    torch.save(state, os.path.join(save_model_path, "checkpoint.pth.tar"))
                else:
                    torch.save(state, os.path.join(save_model_path, "checkpoint.pth.tar"))
                    torch.save(state, os.path.join(save_model_path, "model_best.pth.tar"))

        # one stage end, we should change the operations num (divided 2)
        weight_down=F.softmax(model.module.arch_parameters()[0], dim=-1).data.cpu().numpy()
        weight_up=F.softmax(model.module.arch_parameters()[1], dim=-1).data.cpu().numpy()
        weight_normal=F.softmax(model.module.arch_parameters()[2], dim=-1).data.cpu().numpy()
        weight_network=F.softmax(model.module.arch_parameters()[3], dim=-1).data.cpu().numpy()
        logger.info("alphas_down: \n{}".format(weight_down))
        logger.info("alphas_up: \n{}".format(weight_up))
        logger.info("alphas_normal: \n{}".format(weight_normal))
        logger.info("alphas_network: \n{}".format(weight_network))
        genotype = model.module.genotype()
        _, (_vmr, _vms, _vmp, _vmf, _vmjc, _vmd, _vmacc) = infer(args, model, test_dataloader, criterion)
        sp_recoder_list.append((sp_epoch-1,genotype,_vmd,_vmjc))
        with open(os.path.join(args.save_path,"stage{}_recoder_list.pkl".format(sp)),"wb") as re:
            pickle.dump(sp_recoder_list,re)

        logger.info('Stage:{} \n  Genotype: {}'.format(sp,genotype))
        logger.info('------Stage {} end ! Then  Dropping Paths------'.format(sp))

        # 6                4              7
        # CellLinkDownPos CellLinkUpPos CellPos
        # # 6-->3-->1
        # drop_op_down = [3, 2]
        # # 4-->2-->1
        # drop_op_up = [2, 1]
        # # 7-->4-->1
        # drop_op_normal = [3, 3]
        # update switches in 0 stage end
        if sp==0:
            switches_down=update_switches(weight_down.copy(),switches_down.copy(),CellLinkDownPos,drop_op_down[sp])
            switches_up=update_switches(weight_up.copy(),switches_up.copy(),CellLinkUpPos,drop_op_up[sp])
            switches_normal=update_switches(weight_normal.copy(),switches_normal.copy(),CellPos,drop_op_normal[sp])
            logger.info('switches_down = %s', switches_down)
            logger.info('switches_up = %s', switches_up)
            logger.info('switches_normal = %s', switches_normal)
            logging_switches(logger,switches_down,CellLinkDownPos)
            logging_switches(logger,switches_up,CellLinkUpPos)
            logging_switches(logger,switches_normal,CellPos)
        else:
            # sp==1 is the final stage, we don`t need the keep operations
            # because we has the model.genotype
            # show the final one op in 14 mixop
            switches_down=update_switches(weight_down.copy(),switches_down.copy(),CellLinkDownPos,drop_op_down[sp])
            switches_up=update_switches(weight_up.copy(),switches_up.copy(),CellLinkUpPos,drop_op_up[sp])
            switches_normal=update_switches_nozero(weight_normal.copy(),switches_normal.copy(),CellPos,drop_op_normal[sp])
            logger.info('switches_down = %s', switches_down)
            logger.info('switches_up = %s', switches_up)
            logger.info('switches_normal = %s', switches_normal)
            logging_switches(logger,switches_down,CellLinkDownPos)
            logging_switches(logger,switches_up,CellLinkUpPos)
            logging_switches(logger,switches_normal,CellPos)
    writer.close()


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha, 4)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index1 = torch.randperm(batch_size).cuda()
        index2 = torch.randperm(batch_size).cuda()
        index3 = torch.randperm(batch_size).cuda()
    else:
        index1 = torch.randperm(batch_size)
        index2 = torch.randperm(batch_size)
        index3 = torch.randperm(batch_size)

    lam = softmax(lam)
    mixed_x = lam[0] * x + lam[1] * x[index1, :] + lam[2] * x[index2, :] + lam[3] * x[index3, :]
    y_a, y_b, y_c , y_d = y, y[index1], y[index2], y[index3]
    return mixed_x, y_a, y_b, y_c, y_d, lam


def train(args, train_queue, val_queue, model, criterion, optimizer_weight, optimizer_arch, train_arch):
    Train_recoder = BinaryIndicatorsMetric()
    w_loss_recoder = AverageMeter()
    a_loss_recoder = AverageMeter()
    model.train()
    for step, (input, target,_) in tqdm(enumerate(train_queue)):
        input = input.to(args.device)
        target = target.to(args.device)
        # alpha=1 open mixup   alpha=0 close mixup
        mixup_images,target,perm_target1,perm_target2,perm_target3,lam=mixup_data(input,target,alpha=args.alpha,use_cuda=args.use_cuda)
        # input is B C H W   target is B,1,H,W  preds: B,1,H,W
        optimizer_weight.zero_grad()
        preds = model(mixup_images)
        assert isinstance(preds, list)
        preds = [pred.view(pred.size(0), -1) for pred in preds]
        target = target.view(target.size(0), -1)
        perm_target1=perm_target1.view(perm_target1.size(0), -1)
        perm_target2=perm_target2.view(perm_target2.size(0), -1)
        perm_target3=perm_target3.view(perm_target3.size(0), -1)
        torch.cuda.empty_cache()

        if args.deepsupervision:
            for i in range(len(preds)):
                if i == 0:
                    target1_loss = criterion(preds[i], target)
                target1_loss += criterion(preds[i], target)
        else:
            target1_loss = criterion(preds[-1], target)
        if args.deepsupervision:
            for i in range(len(preds)):
                if i == 0:
                    target2_loss = criterion(preds[i], perm_target1)
                target2_loss += criterion(preds[i], perm_target1)
        else:
            target2_loss = criterion(preds[-1], perm_target1)
        if args.deepsupervision:
            for i in range(len(preds)):
                if i == 0:
                    target3_loss = criterion(preds[i], perm_target2)
                target3_loss += criterion(preds[i], perm_target2)
        else:
            target3_loss = criterion(preds[-1], perm_target2)

        if args.deepsupervision:
            for i in range(len(preds)):
                if i == 0:
                    target4_loss = criterion(preds[i], perm_target3)
                target4_loss += criterion(preds[i], perm_target3)
        else:
            target4_loss = criterion(preds[-1], perm_target3)

        w_loss=lam[0]*target1_loss+lam[1]*target2_loss+lam[2]*target3_loss+lam[3]*target4_loss
        w_loss.backward()
        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.weight_parameters(), args.grad_clip)
        optimizer_weight.step()
        w_loss_recoder.update(w_loss.item(), 1)
        # get all the indicators
        if (step + 1) % args.compute_freq == 0:
            Train_recoder.update(labels=target, preds=preds[-1], n=1)

        # update network arch parameters
        if train_arch:
            # In the original implementation of DARTS, it is input_search, target_search = next(iter(valid_queue), which slows down
            # the training when using PyTorch 0.4 and above.
            try:
                input_search, target_search,_ = next(valid_queue_iter)
            except:
                valid_queue_iter = iter(val_queue)
                input_search, target_search,_ = next(valid_queue_iter)
            input_search = input_search.to(args.device)
            target_search = target_search.to(args.device)
            mixup_input_search, target_search, perm_target_search1, perm_target_search2, perm_target_search3, alam = mixup_data(input_search, target_search, alpha=args.alpha, use_cuda=args.use_cuda)
            optimizer_arch.zero_grad()

            archs_preds = model(mixup_input_search)
            archs_preds = [pred.view(pred.size(0), -1) for pred in archs_preds]
            target_search = target_search.view(target_search.size(0), -1)
            perm_target_search1=perm_target_search1.view(perm_target_search1.size(0), -1)
            perm_target_search2=perm_target_search2.view(perm_target_search2.size(0), -1)
            perm_target_search3=perm_target_search3.view(perm_target_search3.size(0), -1)
            torch.cuda.empty_cache()

            if args.deepsupervision:
                for i in range(len(archs_preds)):
                    if i == 0:
                        a_loss1 = criterion(archs_preds[i], target_search)
                    a_loss1 += criterion(archs_preds[i], target_search)
            else:
                a_loss1 = criterion(archs_preds[-1], target_search)
            if args.deepsupervision:
                for i in range(len(archs_preds)):
                    if i == 0:
                        a_loss2 = criterion(archs_preds[i], perm_target_search1)
                    a_loss2 += criterion(archs_preds[i], perm_target_search1)
            else:
                a_loss2 = criterion(archs_preds[-1], perm_target_search1)
            if args.deepsupervision:
                for i in range(len(archs_preds)):
                    if i == 0:
                        a_loss3 = criterion(archs_preds[i], perm_target_search2)
                    a_loss3 += criterion(archs_preds[i], perm_target_search2)
            else:
                a_loss3 = criterion(archs_preds[-1], perm_target_search2)
            if args.deepsupervision:
                for i in range(len(archs_preds)):
                    if i == 0:
                        a_loss4 = criterion(archs_preds[i], perm_target_search3)
                    a_loss4 += criterion(archs_preds[i], perm_target_search3)
            else:
                a_loss4 = criterion(archs_preds[-1], perm_target_search3)

            a_loss=alam[0]*a_loss1+alam[1]*a_loss2+alam[2]*a_loss3+alam[3]*a_loss4
            a_loss.backward()
            if args.grad_clip:
                nn.utils.clip_grad_norm_(model.arch_parameters(), args.grad_clip)
            optimizer_arch.step()
            a_loss_recoder.update(a_loss.item(), 1)

    weight_loss_avg = w_loss_recoder.avg
    if train_arch:
        arch_loss_avg = a_loss_recoder.avg
    else:
        arch_loss_avg = 0
    mr, ms, mp, mf, mjc, md, macc = Train_recoder.get_avg
    return weight_loss_avg, arch_loss_avg, mr, ms, mp, mf, mjc, md, macc



def infer(args, model, val_queue, criterion):
    OtherVal = BinaryIndicatorsMetric()
    val_loss = AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target,_) in tqdm(enumerate(val_queue)):
            input = input.to(args.device)
            target = target.to(args.device)
            preds = model(input)
            preds = [pred.view(pred.size(0), -1) for pred in preds]
            target = target.view(target.size(0), -1)
            if args.deepsupervision:
                for i in range(len(preds)):
                    if i == 0:
                        loss = criterion(preds[i], target)
                    loss += criterion(preds[i], target)
            else:
                loss = criterion(preds[-1], target)
            val_loss.update(loss.item(), 1)
            OtherVal.update(labels=target, preds=preds[-1], n=1)
        return val_loss.avg, OtherVal.get_avg



def update_switches(weight,switches,PRIMITIVES,drop_nums):
    assert len(weight[0])>drop_nums and len(switches)==len(weight) and len(PRIMITIVES)==len(switches[0])
    for i in range(len(switches)):
        idxs = []
        for j in range(len(PRIMITIVES)):
            if switches[i][j]:
                idxs.append(j)
        drop=np.argsort(weight[i])[:drop_nums]
        for idx in drop:
            switches[i][idxs[idx]] = False
    return switches


def update_switches_nozero(weight, switches, PRIMITIVES, drop_nums):
    assert len(weight[0]) > drop_nums and len(switches) == len(weight) and len(PRIMITIVES) == len(switches[0])
    for i in range(len(switches)):
        mixop_array=weight[i]
        keep_ops = []
        for j in range(len(PRIMITIVES)):
            if switches[i][j]:
                keep_ops.append(j)
        if switches[i][0]:
            mixop_array[0]=0
        max_index=int(np.argmax(mixop_array))
        keep_index=keep_ops[max_index]
        for j in range(len(switches[i])):
            switches[i][j]=False
        switches[i][keep_index]=True
    return switches


def get_min_k(input_in, k):
    input = copy.deepcopy(input_in)
    index = []
    for i in range(k):
        idx = np.argmin(input)
        index.append(idx)
        input[idx] = 1
    return index


def get_min_k_no_zero(w_in, idxs, k):
    w = copy.deepcopy(w_in)
    index = []
    if 0 in idxs:
        zf = True
    else:
        zf = False
    if zf:
        w = w[1:]
        index.append(0)
        k = k - 1
    for i in range(k):
        idx = np.argmin(w)
        w[idx] = 1
        if zf:
            idx = idx + 1
        index.append(idx)
    return index

def logging_switches(logging,switches,PRIMITIVES):
    for i in range(len(switches)):
        ops = []
        for j in range(len(switches[i])):
            if switches[i][j]:
                ops.append(PRIMITIVES[j])
        logging.info(ops)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unet serieas Search')
    # Add default argument
    parser.add_argument('--model', type=str, default='Nas_Search_Unet',
                        help='Model to train and evaluation')
    parser.add_argument('--note',type=str,default='_',help="folder name note")
    parser.add_argument('--dataset', type=str, default='total',
                        help='Model to train and evaluation')
    parser.add_argument('--base_size', type=int, default=256, help="resize base size")
    parser.add_argument('--crop_size', type=int, default=256, help="crop  size")
    parser.add_argument('--epochs', type=int, default=80, help="search epochs")
    parser.add_argument('--train_batch', type=int, default=6, help="train_batch")
    parser.add_argument('--val_batch', type=int, default=6, help="val_batch ")
    parser.add_argument('--num_workers', type=int, default=4, help="dataloader numworkers")
    parser.add_argument('--train_portion', type=float, default=0.5, help="dataloader numworkers")

    # search network setting
    parser.add_argument('--num_classes', type=int, default=1, help="output feature channel")
    parser.add_argument('--input_c', type=int, default=3, help="input img channel")
    parser.add_argument('--init_channel', type=int, default=16, help="init_channel for first leavel search cell")
    parser.add_argument('--meta_node_num', type=int, default=4, help="middle_nodes")
    parser.add_argument('--layers', type=int, default=7, help="layers")
    parser.add_argument('--use_sharing', type=bool, default=True,
                        help="The down op and up op have same normal operations")
    parser.add_argument('--depth', type=int, default=4, help="UnetFabrics`s layers and depth ")
    parser.add_argument('--double_down_channel', action='store_true', help="double_down_channel")
    parser.add_argument('--dropout_prob', type=float, default=0, help="dropout_prob")
    parser.add_argument('--use_softmax_head', type=bool, default=False, help='use_softmax_head')
    parser.add_argument('--alpha', type=float, default=0.5, help=">0 mixup  =0 nomixup")

    # model and device setting
    parser.add_argument('--init_weight_type', type=str, default="kaiming", help="the model init ")
    parser.add_argument('--arch_after', type=int, default=10,
                        help=" the first arch_after epochs without arch parameters traing")
    parser.add_argument('--infer_epoch', type=int, default=4, help=" val freq(epoch) ")
    parser.add_argument('--compute_freq', type=int, default=40, help=" compute freq(epoch) ")
    parser.add_argument('--gpus', type=int, default=1, help=" use cuda or not ")
    parser.add_argument('--grad_clip', type=int, default=0, help=" grid clip to ignore grad boom")
    parser.add_argument('--manualSeed', type=int, default=100, help=" manualSeed ")
    # seatch setting
    parser.add_argument('--loss', type=str, choices=['bce', 'bcelog', 'dice', 'softdice'],
                        default="bcelog", help="loss name ")
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adm'], default='sgd',
                        help=" model_optimizer ! ")
    parser.add_argument('--deepsupervision', action='store_true', help=" deepsupervision nas uent ")

    # lr
    parser.add_argument('--lr', type=float, default=0.025, help="weight parameters lr ")
    parser.add_argument('--lr_min', type=float, default=1e-5, help=" min arch parameters lr  ")
    parser.add_argument('--weight_decay', type=float, default=3e-4, help=" for weight parameters lr  ")
    parser.add_argument('--arch_lr', type=float, default=2e-4, help="arch parameters lr ")
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help=" for arch parameters lr ")
    parser.add_argument('--momentum', type=float, default=0.9, help=" momentum  ")
    # resume
    parser.add_argument('--resume', type=str, default=None, help=" resume file path")
    args = parser.parse_args()
    main(args)

