#coding:utf-8
# baseline
import argparse
from torch.nn import init
from .unet import U_Net
from .unetplusplus import NestedUNet
from .attention_unet import AttU_Net_v1,AttU_Net_v2
from .attention_r2unet import R2AttU_Net_v1,R2AttU_Net_v2
from .r2unet import R2U_Net
from .multires_unet import MultiResUnet
# nas unet
from .nas_unet import NAS_Unet


models_dict={
    'unet':U_Net,
    'attention_unet_v1':AttU_Net_v1,
    'attention_unet_v2':AttU_Net_v2,
    'unet++':NestedUNet,
    'r2unet':R2U_Net,
    'att_r2unet_v1':R2AttU_Net_v1,
    'att_r2unet_v2': R2AttU_Net_v2,
    'multires_unet':MultiResUnet
}

def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)


def get_models(args):
    '''get the correct model '''
    model_name=args.model
    if model_name=="unet":
        model=models_dict[model_name](args.im_channel,args.class_num)
    elif model_name=="unet++":
        model=models_dict[model_name](args.im_channel,args.class_num,args.deepsupervision)
    elif model_name=="r2unet":
        model= models_dict[model_name](args.im_channel,args.class_num,t=args.time_step)
    elif model_name=="multires_unet":
        model= models_dict[model_name](args.im_channel,args.class_num,args.alpha)
    elif model_name=="attention_unet_v1":
        model= models_dict[model_name](args.im_channel,args.class_num)
    elif model_name=="attention_unet_v2":
        model= models_dict[model_name](args.im_channel,args.class_num)
    elif model_name=="att_r2unet_v1":
        model= models_dict[model_name](args.im_channel,args.class_num,t=args.time_step)
    elif model_name=="att_r2unet_v2":
        model= models_dict[model_name](args.im_channel,args.class_num,t=args.time_step)
    else:
        raise  NotImplementedError("the model is not exists !")
    init_weights(model,args.init_weight_type)
    return model






if __name__=="__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.im_channel=3
    args.class_num=2
    args.time_step=2
    args.deepsupervision=False
    args.alpha=1.67
    args.init_weight_type="kaiming"
    name_list=list(models_dict.keys())
    for i in range(len(name_list)):
        args.model=name_list[i]
        model=get_models(args)
        #print(model)



