import threading
import torch
import numpy as np
import torch.nn.functional as F

# include main Segmentation indicators
# must compute each one and .mean
class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scroes"""
    def __init__(self, nclass=3):
        self.nclass = nclass
        self.reset()

    def update(self, labels, preds):
        '''
        predict: input 4D tensor  B,NUM_CLASS,H,W
        target: label 3D tensor   B,H,W
        0 1 2(bk)
        '''

        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    @property
    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        Dice= 2.0 * self.total_inter / (np.spacing(1) + self.total_union+ self.total_inter)
        Dice=Dice.mean()
        return pixAcc, mIoU,Dice


    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        return




class RefugeIndicatorsMetric(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.cup_dice=0.
        self.cup_miou=0.
        self.cup_macc=0.
        self.disc_dice=0.
        self.disc_miou=0.
        self.disc_macc=0.
        self.batch_num=0.

    def update(self,output,target):
        '''
        :param preds:  n,c h,w
        :param target: n,h,w   0,1,2(bk)
        :return:
        '''
        n,c,h,w=output.size()
        output=F.softmax(output,dim=1)
        # n,h,w
        output=torch.max(output,dim=1)[1]
        target=target.cpu().numpy()
        output=output.cpu().numpy()
        T_mask_cup=(target==0).astype(np.uint8)
        T_mask_disc=(target<2).astype(np.uint8)
        O_mask_cup=(output==0).astype(np.uint8)
        O_mask_disc=(output<2).astype(np.uint8)
        # n,h,w
        cup_dice=self.dice_coef(O_mask_cup,T_mask_cup)
        disc_dice=self.dice_coef(O_mask_disc,T_mask_disc)
        self.cup_dice+=cup_dice
        self.disc_dice+=disc_dice

        # total acc of cup and disc
        cup_miou=self.miou(O_mask_cup,T_mask_cup)
        disc_miou=self.miou(O_mask_disc,T_mask_disc)
        self.cup_miou+=cup_miou
        self.disc_miou+=disc_miou

        cup_macc=self.mAcc(O_mask_cup,T_mask_cup)
        disc_macc=self.mAcc(O_mask_disc,T_mask_disc)
        self.cup_macc+=cup_macc
        self.disc_macc+=disc_macc
        self.batch_num+=1

        # miou
    def dice_coef(self,pred,target):
        '''
        :param pred: n h,w
        :param target: n,h,w
        :return:
        '''
        # compute each images`s dice
        smooth = 1e-8
        intersection = np.sum(pred * target,axis=(1,2))
        unionpp=np.sum(target * target,axis=(1,2)) + np.sum(pred * pred,axis=(1,2))
        dice=(2. * intersection + smooth) / (unionpp + smooth)
        #print("Iter:{} unionpp:{}".format(intersection,unionpp))
        return dice.mean()

    def miou(self,pred,target):
        '''
        :param pred:  n,h,w
        :param target:
        :return:
        '''
        smooth = 1e-8
        # each image`s inter
        intersection = np.sum(pred * target,axis=(1,2))
        union=np.sum(target * target,axis=(1,2)) + np.sum(pred * pred,axis=(1,2))-intersection
        assert (intersection <= union).all(), \
            "Intersection area should be smaller than Union area"
        miou=(smooth+intersection)/(smooth+union)
        #print("Iter:{} union:{}".format(intersection,union))
        return miou.mean()

    def mAcc(self,pred,target):
        '''
        :param pred:  input value is 0 or 1
        :param target:
        :return:
        '''
        smooth = 1e-8
        # 0,1 ---1,2
        pred=pred+1
        target=target+1
        label_pixels=np.sum(target>0,axis=(1,2))
        cprrect_pixels=np.sum((pred==target)*(target>0),axis=(1,2))
        assert (cprrect_pixels <= label_pixels).all(), \
            "cprrect_pixels area should be smaller than label_pixels area"
        acc=(cprrect_pixels+smooth)/(label_pixels+smooth)
        #print("label:{} acc:{}".format(label_pixels,cprrect_pixels))
        return acc.mean()

    def avg(self):
        '''
        :return:  return refuge avg acc dice moiu
        '''
        cup=(self.cup_dice/self.batch_num,self.cup_miou/self.batch_num,self.cup_macc/self.batch_num)
        disc=(self.disc_dice/self.batch_num,self.disc_miou/self.batch_num,self.disc_macc/self.batch_num)
        return cup,disc


class BinaryIndicatorsMetric(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.length=0
        self.Recall=0
        self.Specificity=0
        self.Precision=0
        self.F1=0
        self.Jc=0
        self.Dice=0
        self.Acc=0

    def update(self,labels,preds,n=1,threshold=0.5):
        '''
        predict: preds 2D tensor  B,-1    without torch.sigmod
        target: label 2D tensor   B,-1
        '''
        # batch avg score
        assert labels.size(0)==preds.size(0)
        preds=F.sigmoid(preds)
        # B,H,W
        preds = preds > threshold
        labels = labels == torch.max(labels)
        corr = torch.sum(preds == labels)
        tensor_size = labels.size(0)*labels.size(1)
        self.acc = float(corr) / float(tensor_size)

        TP=((preds == 1) + (labels == 1)) == 2
        FN= ((preds == 0) + (labels == 1)) == 2
        FP=((preds == 1) + (labels == 0)) == 2
        TN=((preds == 0) + (labels == 0)) == 2

        self.recall=float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)
        self.specificity=float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)
        self.precision=float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)
        self.f1=2 * self.recall * self.precision / (self.recall + self.precision+ 1e-6)

        Inter = torch.sum((preds + labels) == 2)
        Union = torch.sum((preds + labels) >= 1)
        self.jc=float(Inter) / (float(Union) + 1e-6)
        self.dice=float(2 * Inter) / (float(torch.sum(preds) + torch.sum(labels)) + 1e-6)

        self.Recall+=self.recall
        self.Specificity+=self.specificity
        self.Precision+=self.precision
        self.F1+=self.f1
        self.Jc+=self.jc
        self.Dice+=self.dice
        self.Acc+=self.acc
        self.length+=n

    @property
    def get_current(self):
        return  self.recall,self.specificity,self.precision,self.f1,self.jc,self.dice,self.acc

    @property
    def get_avg(self):
        return self.Recall/self.length,self.Specificity/self.length,\
               self.Precision/self.length,self.F1/self.length,self.Jc/self.length,\
               self.Dice/self.length,self.Acc/self.length




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def batch_pix_accuracy(output, target):
    """Batch Pixel Accuracy
    Args:
        predict: input 4D tensor  B,NUM_CLASS,H,W
        target: label 3D tensor   B,H,W
    """
    output = F.softmax(output, dim=1)
    predict = torch.max(output, 1)[1]
    # label: 0, 1, ..., nclass - 1
    # Note: 0 is background
    # change the acc compute
    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target) * (target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled



# JACCARD
def batch_intersection_union(output, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    output = F.softmax(output, dim=1)
    predict = torch.max(output, 1)[1]
    mini = 1
    maxi = nclass - 1
    nbins = nclass - 1

    # label is: 0, 1, 2, ..., nclass-1
    # Note: 0 is background
    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1
    # available predict area
    predict = predict * (target > 0).astype(predict.dtype)
    # correct predict area
    intersection = predict * (predict == target)

    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union




def dice_coefficient(input, target, smooth=1.0):
    '''
    :param input: input 4D tensor  B,NUM_CLASS,H,W
    :param target: label 3D tensor   B,H,W
    :param smooth:
    :return:
    '''
    assert smooth > 0, 'Smooth must be greater than 0.'
    probs = F.softmax(input, dim=1)
    encoded_target = probs.detach() * 0
    encoded_target.scatter_(1, target.unsqueeze(1), 1)
    encoded_target = encoded_target.float()

    num = probs * encoded_target  # b, c, h, w -- p*g
    num = torch.sum(num, dim=3)  # b, c, h
    num = torch.sum(num, dim=2)  # b, c

    den1 = probs * probs  # b, c, h, w -- p^2
    den1 = torch.sum(den1, dim=3)  # b, c, h
    den1 = torch.sum(den1, dim=2)  # b, c

    den2 = encoded_target * encoded_target  # b, c, h, w -- g^2
    den2 = torch.sum(den2, dim=3)  # b, c, h
    den2 = torch.sum(den2, dim=2)  # b, c

    dice = (2 * num + smooth) / (den1 + den2 + smooth)  # b, c
    # batch avg dice
    return dice.mean().mean()


####################################################################################
def get_boundary(data, img_dim=2, shift=-1):
    data = data > 0
    edge = np.zeros_like(data)
    for nn in range(img_dim):
        edge += ~(data ^ np.roll(~data, shift=shift, axis=nn))
    return edge.astype(int)


def rel_abs_vol_diff(y_true, y_pred):
    return np.abs((y_pred.sum() / y_true.sum() - 1) * 100)




# DICE SCORE
def numpy_dice(y_true, y_pred, axis=None, smooth=1.0):
    intersection = y_true * y_pred
    return (2. * intersection.sum(axis=axis) + smooth) / (
                np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis) + smooth)
# ACC ,MEAN IOU
def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    def _fast_hist(label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        return hist
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


if __name__=="__main__":
    # 读取图片测试
    from PIL import Image
    image_sir=r'C:\Users\rileyliu\Desktop\REFUGE\train_GT\g0001.bmp'
    label=Image.open(image_sir).convert('L')
    label=np.asarray(label)
    label_copy = 255 * np.ones(label.shape, dtype=np.float32)  # 0 disc 128 cup 255 bk
    id_to_trained={0: 0, 128: 1, 255: 2}
    for k, v in id_to_trained.items():
        label_copy[label == k] = v
    # BGR image to inference
    label_copy = torch.from_numpy(label_copy).unsqueeze(0)
    recoder=RefugeIndicatorsMetric()
    recoder.update(label_copy,label_copy)
    cup,disc=recoder.avg()
    print(cup)
    print(disc)
