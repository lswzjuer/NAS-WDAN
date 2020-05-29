from torch.nn import init
import argparse
from .UnetFabrices7 import UnetLayer7
from .UnetFabrices9 import UnetLayer9
from .UnetFabrices9_v2 import UnetLayer9_v2


models_dict={
    'UnetLayer7':UnetLayer7,
    'UnetLayer9':UnetLayer9,
    'UnetLayer9_v2':UnetLayer9_v2
}


def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if classname!="NoneType":
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
            # elif classname.find('BatchNorm2d') != -1:
            #     init.normal_(m.weight.data, 1.0, gain)
            #     init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)


def get_models(args,switches_normal,switches_down,switches_up):
    '''get the correct model '''
    model_name=args.model
    if model_name=="UnetLayer7":
        assert args.layers==7
        model=models_dict[model_name](input_c=args.input_c,c=args.init_channel,num_classes=args.num_classes,
                    meta_node_num=args.meta_node_num, layers=args.layers,dp=args.dropout_prob,
                 use_sharing=args.use_sharing,double_down_channel=args.double_down_channel,use_softmax_head=args.use_softmax_head,
                 switches_normal=switches_normal,switches_down=switches_down,switches_up=switches_up)

    elif model_name=="UnetLayer9":
        assert args.layers==9
        model=models_dict[model_name](input_c=args.input_c,c=args.init_channel,num_classes=args.num_classes,
                    meta_node_num=args.meta_node_num, layers=args.layers,dp=args.dropout_prob,
                 use_sharing=args.use_sharing,double_down_channel=args.double_down_channel,use_softmax_head=args.use_softmax_head,
                 switches_normal=switches_normal,switches_down=switches_down,switches_up=switches_up)
    elif model_name=='UnetLayer9_v2':
        assert args.layers==9
        model=models_dict[model_name](input_c=args.input_c,c=args.init_channel,num_classes=args.num_classes,
                    meta_node_num=args.meta_node_num, layers=args.layers,dp=args.dropout_prob,
                 use_sharing=args.use_sharing,double_down_channel=args.double_down_channel,use_softmax_head=args.use_softmax_head,
                 switches_normal=switches_normal,switches_down=switches_down,switches_up=switches_up)
    else:
        raise  NotImplementedError("the model is not exists !")
    init_weights(model,args.init_weight_type)
    return model


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.num_class=1
    args.im_ch=3
    args.init_channel=16
    args.middle_nodes=4
    args.layers=7
    args.init_weight_type="kaiming"
    args.model="UnetLayer7"
    model=get_models(args)
    print(model)
