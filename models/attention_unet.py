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


class Attention_block(nn.Module):
    def __init__(self, F_g, F_x, F_int):
        '''
        :param F_g:  decoder path channel
        :param F_x:  encoder path channel
        :param F_int: inter_channels
        '''
        super(Attention_block, self).__init__()
        # W_g is the decoder input
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # W_x is the encoder input
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
        )
        self.relu = nn.ReLU(inplace=True)
        self.sigmod=nn.Sigmoid()
    # g is the decoder input and x is the encoder input
    # x.size==2*g.size
    def forward(self, g, x):
        b,c,h,w=x.size()
        assert b==g.size(0)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        g1=F.upsample(input=g1,size=x1.size()[2:],mode='bilinear')
        inter=self.relu(g1+x1)
        inter=self.psi(inter)
        #
        inter=self.sigmod(inter)
        inter=F.upsample(inter,size=x.size()[2:],mode='bilinear')
        inter=inter.expand_as(x)
        return inter*x


class AttU_Net_v1(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net_v1, self).__init__()

        num_filters=[64,128,256,512,1024]
        #num_filters=[32,64,128,256,512]
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=num_filters[0])
        self.Conv2 = conv_block(ch_in=num_filters[0], ch_out=num_filters[1])
        self.Conv3 = conv_block(ch_in=num_filters[1], ch_out=num_filters[2])
        self.Conv4 = conv_block(ch_in=num_filters[2], ch_out=num_filters[3])
        self.Conv5 = conv_block(ch_in=num_filters[3], ch_out=num_filters[4])

        self.Up5 = up_conv(ch_in=num_filters[4], ch_out=num_filters[3])
        self.Att5 = Attention_block(F_g=num_filters[3], F_x=num_filters[3], F_int=num_filters[3]//2)
        self.Up_conv5 = conv_block(ch_in=num_filters[4], ch_out=num_filters[3])

        self.Up4 = up_conv(ch_in=num_filters[3], ch_out=num_filters[2])
        self.Att4 = Attention_block(F_g=num_filters[2], F_x=num_filters[2], F_int=num_filters[2]//2)
        self.Up_conv4 = conv_block(ch_in=num_filters[3], ch_out=num_filters[2])

        self.Up3 = up_conv(ch_in=num_filters[2], ch_out=num_filters[1])
        self.Att3 = Attention_block(F_g=num_filters[1], F_x=num_filters[1], F_int=num_filters[1]//2)
        self.Up_conv3 = conv_block(ch_in=num_filters[2], ch_out=num_filters[1])

        self.Up2 = up_conv(ch_in=num_filters[1], ch_out=num_filters[0])
        self.Att2 = Attention_block(F_g=num_filters[0], F_x=num_filters[0], F_int=num_filters[0]//2)
        self.Up_conv2 = conv_block(ch_in=num_filters[1], ch_out=num_filters[0])

        self.Conv_1x1 = nn.Conv2d(num_filters[0], output_ch, kernel_size=1, stride=1, padding=0)

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
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net_v2(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net_v2, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=1024, F_x=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=512, F_x=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=256, F_x=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=128, F_x=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)


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
        x4 = self.Att5(g=x5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d5, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d4, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d3, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


def calc_parameters_count(model):
    import numpy as np
    return np.sum(np.prod(v.size()) for v in model.parameters())/1e6


if __name__=="__main__":
    model=AttU_Net_v1()
    print(calc_parameters_count(model))
