import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, L1Loss, BCELoss,BCEWithLogitsLoss
import numpy as np
from skimage.measure import label, regionprops
import scipy.ndimage as ndimage
import copy

#Custom loss function
# CLASS torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')
# input (N,*) original output
# target(N,*)

# nn.MSELoss()
# input (N,*) original output
# target(N,*)

# nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean')
# input (N,*) original output
# target(N,*)
# Switch between l1 and l2 to prevent gradient explosion

# nn.LogSoftmax()和nn.NLLLoss() == nn.CrossEntropyloss
# input:  N,c
# target: N





# Binary loss function, input is the probability image which channel is 1, without sigmod func
class BCELoss_v1(BCELoss):
    def __init__(self):
        super(BCELoss_v1, self).__init__()
    def forward(self, input, target):
        '''
        :param input: (N,*), input must be the sigmod func output
        :param target: (N,*) * is any other dims but be the same with input,
        :return:
        '''
        return super().forward(input, target)

class BCEWithLogitsLoss_v1(BCEWithLogitsLoss):
    def __init__(self):
        super(BCEWithLogitsLoss_v1, self).__init__()
    def forward(self, input, target):
        '''
        :param input: (N,*), input must be the original probability image
        :param target: (N,*) * is any other dims but be the same with input,
        :return:  sigmod + BCELoss
        '''
        return super().forward(input, target)

# binary crossentory and dice losss,
class BCEDiceLoss(nn.Module):
    def __init__(self,bce_weight=0.5,dice_weight=1):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight=bce_weight
        self.dice_weight=dice_weight

    def forward(self, input, target):
        '''
        :param input: (N,*), input must be the original probability image
        :param target: (N,*) * is any other dims but be the same with input,
        : shape is  N -1 or  N 1 H W
        :return:  sigmod + BCELoss +  sigmod + DiceLoss
        '''
        # N ample`s average
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return self.bce_weight* bce + self.dice_weight*dice


class MSEWithLogitsLoss(MSELoss):
    """
    This loss combines a `Sigmoid` layer and the `MSELoss` in one single class.
    """

    def __init__(self):
        super(MSEWithLogitsLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, target):
        return super().forward(self.sigmoid(input), target)


# multi-classification
class CrossEntropy2d(nn.Module):
    def __init__(self, size_average=True, ignore_label=500):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target,weight=[1,1,0.1]):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        # not ignore_label area
        if weight is not None:
            weight = torch.FloatTensor(weight).to(predict.device)
        else:
            weight=None
        if self.ignore_label is not None:
            target_mask = (target >= 0)*(target!=self.ignore_label)
        else:
            target_mask = (target >= 0)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        # n h w c
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        target=target.view(-1)
        # has add  log +softmax  func
        loss = F.cross_entropy(predict, target, weight=weight, reduction='mean')
        return loss




# You can use this function when the categories are uneven
class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """
    def __init__(self, weight=None, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
    def forward(self, input, target):
        #input  N * C
        #target N
        class_weights = self._class_weights(input.clone())
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            class_weights = class_weights * weight
        return F.cross_entropy(input, target, weight=class_weights, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights


class DiceLoss(nn.Module):
    """Computes Dice Loss, which just 1 - DiceCoefficient described above.
    Additionally allows per-class weights to be provided.
    """
    def __init__(self, epsilon=1e-5, ignore_index=None, sigmoid_normalization=False,
                 skip_last_target=False):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.ignore_index = ignore_index
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify sigmoid_normalization=False.
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)
        # if True skip the last channel in the target
        self.skip_last_target = skip_last_target

    def forward(self, input, target,weight=[1,1,0.1]):
        # get probabilities from logits
        # input shape N,C,h,w
        # target shape N C,h,w
        input = self.normalization(input)
        if weight is not None:
            weight =torch.FloatTensor(weight).to(input.device)
        else:
            weight = None
        if self.skip_last_target:
            target = target[:, :-1, ...]
        per_channel_dice = compute_per_channel_dice(input, target, epsilon=self.epsilon, ignore_index=self.ignore_index,
                                                    weight=weight)
        # Average the Dice score across all channels/classes
        return torch.mean(1. - per_channel_dice)


def compute_per_channel_dice(input, target, epsilon=1e-5, ignore_index=None, weight=None):
    '''
    :param input:
    :param target:
    :param epsilon:
    :param ignore_index:
    :param weight:
    :return:
    '''
    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"
    # mask ignore_index if present
    if ignore_index is not None:
        mask = target.clone().ne_(ignore_index)
        mask.requires_grad = False
        input = input * mask
        target = target * mask
    # C*N
    input = flatten(input)
    # C*N
    target = flatten(target)
    target = target.float()
    # Compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect
    denominator = (input + target).sum(-1)
    return 2. * intersect / denominator.clamp(min=epsilon)


class MultiClassEntropyDiceLoss(nn.Module):
    def __init__(self,size_average=True,weight=[1,1,0.1],ignore_label=None,dice_weight=10):
        super(MultiClassEntropyDiceLoss, self).__init__()
        # (self, size_average=True, ignore_label=500):
        # input n c h w
        self.cross_entropy=CrossEntropy2d(size_average=size_average,ignore_label=ignore_label)
        #  epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=False,
        #                  skip_last_target=False
        # input N,C ,so we need to reshape the input and target
        self.dice=DiceLoss(ignore_index=None,sigmoid_normalization=False,skip_last_target=False)
        self.weight=weight
        self.dice_weight=dice_weight


    def forward(self, preds,target):
        '''
        :param input:  n c h w tensor
        :param target: n h w  tensor
        :return:
        '''
        target=target.long()
        n,c,h,w=preds.size()
        cross_entropy_loss=self.cross_entropy(preds.clone(),target.clone(),self.weight)
        # target one hot encoding
        # n c h w
        encoded_target = preds.detach() * 0
        encoded_target.scatter_(1, target.unsqueeze(1), 1)
        encoded_target=encoded_target.long()
        encoded_target = encoded_target.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        preds_view = preds.transpose(1, 2).transpose(2, 3).contiguous().view(-1,c)
        dice_loss=self.dice(preds_view,encoded_target,self.weight)
        return cross_entropy_loss+dice_loss*self.dice_weight,cross_entropy_loss,dice_loss


class WeightDiceLoss(nn.Module):
    def __init__(self,size_average=True,weight=[1,1,0.1],ignore_label=None):
        super(WeightDiceLoss, self).__init__()
        # (self, size_average=True, ignore_label=500):
        # input n c h w
        # input N,C ,so we need to reshape the input and target
        self.dice=DiceLoss(ignore_index=None,sigmoid_normalization=False,skip_last_target=False)
        self.weight=weight

    def forward(self, preds,target):
        '''
        :param input:  n c h w tensor
        :param target: n h w  tensor
        :return:
        '''
        target=target.long()
        n,c,h,w=preds.size()
        encoded_target = preds.detach() * 0
        encoded_target.scatter_(1, target.unsqueeze(1), 1)
        encoded_target=encoded_target.long()
        encoded_target = encoded_target.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        preds_view = preds.transpose(1, 2).transpose(2, 3).contiguous().view(-1,c)
        dice_loss=self.dice(preds_view,encoded_target,self.weight)
        return dice_loss




class GeneralizedDiceLoss(nn.Module):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf
    """
    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        # mask ignore_index if present
        if self.ignore_index is not None:
            mask = target.clone().ne_(self.ignore_index)
            mask.requires_grad = False
            input = input * mask
            target = target * mask

        input = flatten(input)
        target = flatten(target)
        target = target.float()
        target_sum = target.sum(-1)
        class_weights = Variable(1. / (target_sum * target_sum).clamp(min=self.epsilon), requires_grad=False)
        intersect = (input * target).sum(-1) * class_weights
        if self.weight is not None:
            weight = torch.FloatTensor(self.weight).to(input.device)
            intersect = weight * intersect
        intersect = intersect.sum()
        denominator = ((input + target).sum(-1) * class_weights).sum()
        return 1. - 2. * intersect / denominator.clamp(min=self.epsilon)


class SoftDiceLoss(nn.Module):
    def __init__(self,epsilon=1e-5,weight=None,optimize_bg=False,reduce=True):
        super(SoftDiceLoss, self).__init__()
        self.smooth=epsilon
        self.weight=weight
        self.ig_bg=optimize_bg
        self.reduce=reduce
    def forward(self, input,target):
        '''
        :param input:  N C H W
        :param target: N H W
        :return:
        '''
        def dice_coefficient(input, target, smooth=1.0):
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
            return dice
        dice = dice_coefficient(input, target, smooth=self.smooth)
        if not self.ig_bg:
            dice = dice[:, 1:]  # we ignore bg dice val, and take the fg
        if not type(self.weight) is type(None):
            if not self.ig_bg:
                weight = self.weight[1:]  # ignore bg weight
            weight = weight.size(0) * weight / weight.sum()  # normalize fg weights
            dice = dice * weight  # weighting
        dice_loss = 1 - dice.mean(1)  # loss is calculated using mean over dice vals (n,c) -> (n)
        if self.reduce:
            return dice_loss.mean()
        return dice_loss.sum()

class EdgeWeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, weight=None, ignore_index=-1):
        super(EdgeWeightedCrossEntropyLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        n,c,z,h,w = input[0].shape
        class_weights = self._class_weights(input,target)
        #target = target.view(n,1,z,h,w).repeat(1,c,1,1,1)
        mask = target.float()
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            class_weights = class_weights * weight
        #return F.cross_entropy(input, target, weight=class_weights, ignore_index=self.ignore_index)
        if len(input)==1:
            return F.binary_cross_entropy(input, target, weight=class_weights)
        else:
            loss = torch.zeros(1).cuda()
            for i in range(len(input)):
                weight_bck, weight_obj = self._calculate_maps(mask, class_weights)
                logit = torch.softmax(input[i], dim=1)
                logit = logit.clamp(1e-20, 1. - 1e-20)
                weight_sum = weight_bck + weight_obj
                loss += (-1 * weight_bck * torch.log(logit[:, 0, :, :, :]) - weight_obj * torch.log(logit[:, 1, :, :, :])).sum()/weight_sum.sum()
            return loss

    @staticmethod
    def _class_weights(input, Target):
        # normalize the input first
        input = F.softmax(input[-1], _stacklevel=5)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights1 = Variable(nominator / denominator, requires_grad=False)
        weight_cof = 30
        targ = copy.deepcopy(Target.data)
        #target = Target.detach().clone()
        labeled, label_num = label(targ[0,:,:,:].cpu().numpy(), neighbors=4, background=0, return_num=True)
        image_props = regionprops(labeled, cache=False)
        dis_trf = ndimage.distance_transform_edt(targ[0,:,:,:].cpu().numpy())
        adaptive_obj_dis_weight = np.zeros(targ[0].shape, dtype=np.float32)
        adaptive_obj_dis_weight = adaptive_obj_dis_weight + targ[0,:,:,:].cpu().numpy() * weight_cof
        adaptive_bck_dis_weight = np.ones(targ[0].shape, dtype=np.float32)
        for num in range(1, label_num + 1):
            image_prop = image_props[num - 1]
            bool_dis = np.zeros(image_prop.image.shape)
            bool_dis[image_prop.image] = 1.0
            (min_row, min_col, min_z, max_row, max_col, max_z) = image_prop.bbox
            temp_dis = dis_trf[min_row: max_row, min_col: max_col, min_z:max_z] * bool_dis
            adaptive_obj_dis_weight[min_row: max_row, min_col: max_col, min_z:max_z] = adaptive_obj_dis_weight[min_row: max_row, min_col: max_col, min_z:max_z] + get_obj_dis_weight(temp_dis) * bool_dis
            adaptive_bck_dis_weight[min_row: max_row, min_col: max_col, min_z:max_z] = adaptive_bck_dis_weight[min_row: max_row, min_col: max_col, min_z:max_z] + get_bck_dis_weight(temp_dis) * bool_dis

        adaptive_obj_dis_weight = adaptive_obj_dis_weight[np.newaxis,np.newaxis,:,:,:]
        adaptive_bck_dis_weight = adaptive_bck_dis_weight[np.newaxis,np.newaxis,:,:,:]
        adaptive_dis_weight = np.concatenate((adaptive_bck_dis_weight, adaptive_obj_dis_weight), axis=1)
        adaptive_dis_weight = torch.tensor(adaptive_dis_weight, device=0).float()
        #class_weights = class_weights1[0]+adaptive_bck_dis_weight
        #class_weights = class_weights1[1]+adaptive_obj_dis_weight

        return adaptive_dis_weight #class_weights

    @staticmethod
    def _calculate_maps(mask, weight_maps):
        weight_bck = torch.zeros_like(mask)
        weight_obj = torch.zeros_like(mask)
        temp_weight_bck = weight_maps[:, 0, :, :, :]
        temp_weight_obj = weight_maps[:, 1, :, :, :]
        weight_obj[temp_weight_obj >= temp_weight_bck] = temp_weight_obj[temp_weight_obj >= temp_weight_bck]
        weight_obj = mask * weight_obj
        weight_bck[weight_obj <= temp_weight_bck] = temp_weight_bck[weight_obj <= temp_weight_bck]
        return weight_bck, weight_obj

def get_bck_dis_weight(dis_map, w0=10, eps=1e-20):
    """
    Obtain background (inside grain) weight map
    """
    max_dis = np.amax(dis_map)
    std = max_dis / 2.58 + eps
    weight_matrix = w0 * np.exp(-1 * pow((max_dis - dis_map), 2) / (2 * pow(std, 2)))
    return weight_matrix

def get_obj_dis_weight(dis_map, w0=10, eps=1e-20):
    """
    Obtain a foreground (grain boundary) weight map based on a normal distribution curve with a probability density of 99% at [-2.58*sigma, 2.58*sigma]
  So each time you get the maximum value max_dis, and then calculate sigma = max_dis / 2.58
  finally calculate Loss based on the original paper of U-Net
    """
    max_dis = np.amax(dis_map)
    std = max_dis / 2.58 + eps
    weight_matrix = w0 * np.exp(-1 * pow(dis_map, 2) / (2 * pow(std, 2)))
    return weight_matrix

class BCELossWrapper:
    """
    Wrapper around BCE loss functions allowing to pass 'ignore_index' as well as 'skip_last_target' option.
    """

    def __init__(self, loss_criterion, ignore_index=-1, skip_last_target=False):
        if hasattr(loss_criterion, 'ignore_index'):
            raise RuntimeError(f"Cannot wrap {type(loss_criterion)}. Use 'ignore_index' attribute instead")
        self.loss_criterion = loss_criterion
        self.ignore_index = ignore_index
        self.skip_last_target = skip_last_target

    def __call__(self, input, target):
        if self.skip_last_target:
            target = target[:, :-1, ...]

        assert input.size() == target.size()

        masked_input = input
        masked_target = target
        if self.ignore_index is not None:
            mask = target.clone().ne_(self.ignore_index)
            mask.requires_grad = False

            masked_input = input * mask
            masked_target = target * mask

        return self.loss_criterion(masked_input, masked_target)


class PixelWiseCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=None):
        super(PixelWiseCrossEntropyLoss, self).__init__()
        self.register_buffer('class_weights', class_weights)
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target, weights):
        assert target.size() == weights.size()
        # normalize the input
        log_probabilities = self.log_softmax(input)
        # standard CrossEntropyLoss requires the target to be (NxDxHxW), so we need to expand it to (NxCxDxHxW)
        target = expand_as_one_hot(target, C=input.size()[1], ignore_index=self.ignore_index)
        # expand weights
        weights = weights.unsqueeze(0)
        weights = weights.expand_as(input)

        # mask ignore_index if present
        if self.ignore_index is not None:
            mask = Variable(target.data.ne(self.ignore_index).float(), requires_grad=False)
            log_probabilities = log_probabilities * mask
            target = target * mask

        # create default class_weights if None
        if self.class_weights is None:
            class_weights = torch.ones(input.size()[1]).float().to(input.device)
            self.register_buffer('class_weights', class_weights)

        # resize class_weights to be broadcastable into the weights
        class_weights = self.class_weights.view(1, -1, 1, 1, 1)

        # multiply weights tensor by class weights
        weights = class_weights * weights

        # compute the losses
        result = -weights * target * log_probabilities
        # average the losses
        return result.mean()

class TagsAngularLoss(nn.Module):
    def __init__(self, tags_coefficients):
        super(TagsAngularLoss, self).__init__()
        self.tags_coefficients = tags_coefficients

    def forward(self, inputs, targets, weight):
        assert isinstance(inputs, list)
        # if there is just one output head the 'inputs' is going to be a singleton list [tensor]
        # and 'targets' is just going to be a tensor (that's how the HDF5Dataloader works)
        # so wrap targets in a list in this case
        if len(inputs) == 1:
            targets = [targets]
        assert len(inputs) == len(targets) == len(self.tags_coefficients)
        loss = 0
        for input, target, alpha in zip(inputs, targets, self.tags_coefficients):
            loss += alpha * square_angular_loss(input, target, weight)
        return loss


def square_angular_loss(input, target, weights=None):
    """
    Computes square angular loss between input and target directions.
    Makes sure that the input and target directions are normalized so that torch.acos would not produce NaNs.
    :param input: 5D input tensor (NCDHW)
    :param target: 5D target tensor (NCDHW)
    :param weights: 3D weight tensor in order to balance different instance sizes
    :return: per pixel weighted sum of squared angular losses
    """
    assert input.size() == target.size()
    # normalize and multiply by the stability_coeff in order to prevent NaN results from torch.acos
    stability_coeff = 0.999999
    input = input / torch.norm(input, p=2, dim=1).detach().clamp(min=1e-8) * stability_coeff
    target = target / torch.norm(target, p=2, dim=1).detach().clamp(min=1e-8) * stability_coeff
    # compute cosine map
    cosines = (input * target).sum(dim=1)
    error_radians = torch.acos(cosines)
    if weights is not None:
        return (error_radians * error_radians * weights).sum()
    else:
        return (error_radians * error_radians).sum()

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)

def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """
    assert input.dim() == 3

    shape = input.size()
    shape = list(shape)
    shape.insert(1, C)
    shape = tuple(shape)

    # expand the input tensor to Nx1xDxHxW
    src = input.unsqueeze(1)

    if ignore_index is not None:
        # create ignore_index mask for the result
        expanded_src = src.expand(shape)
        mask = expanded_src == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        src = src.clone()
        src[src == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, src, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, src, 1)



if __name__=="__main__":
    predits=torch.randn(size=(1,3,5,5))
    label=torch.zeros(size=(1,5,5))
    for i in range(2,4):
        for j in range(2,4):
            label[:,i,j]=1
    for i in range(0,2):
        for j in range(0,2):
            label[:,i,j]=2
    predits_output=F.softmax(predits,dim=1)

    loss=MultiClassEntropyDiceLoss(weight=[1,1,0.1])
    loss_res=loss(predits,label)
    print(loss_res)
    
    
    
    