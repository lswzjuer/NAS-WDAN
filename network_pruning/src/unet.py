

import torch.nn as nn
import torch
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()
        nb_filter = [64, 128, 256, 512,1024]
        #nb_filter = [32,64, 128, 256, 512]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=nb_filter[0])
        self.Conv2 = conv_block(ch_in=nb_filter[0], ch_out=nb_filter[1])
        self.Conv3 = conv_block(ch_in=nb_filter[1], ch_out=nb_filter[2])
        self.Conv4 = conv_block(ch_in=nb_filter[2], ch_out=nb_filter[3])
        self.Conv5 = conv_block(ch_in=nb_filter[3], ch_out=nb_filter[4])

        self.Up5 = up_conv(ch_in=nb_filter[4], ch_out=nb_filter[3])
        self.Up_conv5 = conv_block(ch_in=nb_filter[4], ch_out=nb_filter[3])

        self.Up4 = up_conv(ch_in=nb_filter[3], ch_out=nb_filter[2])
        self.Up_conv4 = conv_block(ch_in=nb_filter[3], ch_out=nb_filter[2])

        self.Up3 = up_conv(ch_in=nb_filter[2], ch_out=nb_filter[1])
        self.Up_conv3 = conv_block(ch_in=nb_filter[2], ch_out=nb_filter[1])

        self.Up2 = up_conv(ch_in=nb_filter[1], ch_out=nb_filter[0])
        self.Up_conv2 = conv_block(ch_in=nb_filter[1], ch_out=nb_filter[0])

        self.Conv_1x1 = nn.Conv2d(nb_filter[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1



def calc_parameters_count(model):
    import numpy as np
    return np.sum(np.prod(v.size()) for v in model.parameters())/1e6


if __name__=="__main__":
    model=U_Net()
    print(calc_parameters_count(model))
    for name,module in model.named_modules():
        print(name,type(module).__name__)





# if __name__=="__main__":
#     from torch.nn import init
#     def init_weights(net, init_type='kaiming', gain=0.02):
#         def init_func(m):
#             classname = m.__class__.__name__
#             if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
#                 if init_type == 'normal':
#                     init.normal_(m.weight.data, 0.0, gain)
#                 elif init_type == 'xavier':
#                     init.xavier_normal_(m.weight.data, gain=gain)
#                 elif init_type == 'kaiming':
#                     init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#                 elif init_type == 'orthogonal':
#                     init.orthogonal_(m.weight.data, gain=gain)
#                 else:
#                     raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
#                 if hasattr(m, 'bias') and m.bias is not None:
#                     init.constant_(m.bias.data, 0.0)
#             elif classname.find('BatchNorm2d') != -1:
#                 init.normal_(m.weight.data, 1.0, gain)
#                 init.constant_(m.bias.data, 0.0)
#
#         print('initialize network with %s' % init_type)
#         net.apply(init_func)
#     model=U_Net(3,1)
#     for name,parameter in model.named_parameters():
#         print(parameter)
#         break
#     init_weights(model,init_type="normal")
#     for name, parameter in model.named_parameters():
#         print(parameter)
#         break
#
#     for name,module in model.named_modules():
#         if isinstance(module,nn.Conv2d):
#             print(name)
#             print(module.__class__.__name__)
#
