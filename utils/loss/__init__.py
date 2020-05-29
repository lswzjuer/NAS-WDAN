from .loss import SegmentationLosses

loss_dict={
    'seg':SegmentationLosses,
}

def get_criterion(args,type="seg"):
    ''' get the correct loss '''
    if type=="seg":
        assert args.loss_name in ['dice_loss','cross_entropy','cross_entropy_with_dice']
        return loss_dict["seg"](name=args.loss_name,aux_weight=args.aux_weight)
    else:
        raise NotImplementedError("The loss type is not exist ")


