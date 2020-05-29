import torch.nn as nn
import torch
import torch.nn.functional as F


class conv_bn(nn.Module):
    def __init__(self, ch_in, ch_out,kernel_size=3,stride=1,padding=1):
        super(conv_bn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(ch_out),
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class MultiResBlock(nn.Module):
    ''' multires block'''
    def __init__(self,inchannels,stage_channel,alpha=1.67):
        super(MultiResBlock, self).__init__()
        self.in_channels=inchannels
        self.stage_channels=stage_channel
        self.alpha_channels=alpha*stage_channel
        # three block ,channel is  int(W * 0.167) + int(W * 0.333),int(W * 0.5)
        self.true_channels=int(self.alpha_channels * 0.167) + \
                           int(self.alpha_channels * 0.333)+int(self.alpha_channels * 0.5)

        self.shortcut=conv_bn(self.in_channels,self.true_channels,kernel_size=1,stride=1,padding=0)

        self.conv3x3_1=conv_bn(self.in_channels,int(self.alpha_channels * 0.167),
                               kernel_size=3,stride=1,padding=1)

        self.conv3x3_2=conv_bn(int(self.alpha_channels * 0.167),int(self.alpha_channels *  0.333),
                               kernel_size=3,stride=1,padding=1)

        self.conv3x3_3=conv_bn(int(self.alpha_channels * 0.333),int(self.alpha_channels * 0.5),
                               kernel_size=3,stride=1,padding=1)
        self.concatbn=nn.BatchNorm2d(self.true_channels)

    def forward(self, input):

        shortcut=self.shortcut(input)

        conv1=self.conv3x3_1(input)
        conv1=F.relu(conv1,inplace=True)

        conv2=self.conv3x3_2(conv1)
        conv2=F.relu(conv2,inplace=True)

        conv3=self.conv3x3_3(conv2)
        conv3=F.relu(conv3)

        # conv_concat=torch.cat([conv1,conv2,conv3],dim=1)
        # sum=shortcut+conv_concat
        # sum=self.concatbn(sum)
        # sum=F.relu(sum)

        conv_concat=torch.cat([conv1,conv2,conv3],dim=1)
        conv_concat=self.concatbn(conv_concat)
        sum=shortcut+conv_concat
        sum=F.relu(sum)
        return sum


class RespathConv(nn.Module):
    ''' rea path conv connect'''
    def __init__(self,inchannels,outchannels,length):
        super(RespathConv, self).__init__()
        self.in_channels=inchannels
        self.out_channels=outchannels
        self.block_nums=length
        self.conv1x1=[]
        self.conv3x3=[]
        if self.block_nums>=1:
            self.conv1x1.append(conv_bn(self.in_channels,self.out_channels,kernel_size=1,stride=1,padding=0))
            self.conv3x3.append(conv_bn(self.in_channels,self.out_channels,kernel_size=3,stride=1,padding=1))
            for i in range(self.block_nums-1):
                self.conv1x1.append(conv_bn(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0))
                self.conv3x3.append(conv_bn(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1))
        self.conv1x1=nn.ModuleList(self.conv1x1)
        self.conv3x3=nn.ModuleList(self.conv3x3)

    def forward(self, input):
        if self.block_nums>=1:
            for i in range(self.block_nums):
                shortcut=self.conv1x1[i](input)
                conv=self.conv3x3[i](input)
                sum=shortcut+conv
                input=F.relu(sum,inplace=False)
        return input


# just half the h and w , don`t change channels
class  Downsample(nn.Module):
    def __init__(self,inchannel,outchannel,max_flag=True,trainable=False):
        self.trainable=trainable
        if self.trainable:
            self.dowmsample=nn.Conv2d(inchannel,outchannel,kernel_size=2,stride=2)
        else:
            if max_flag:
                self.dowmsample=nn.MaxPool2d(kernel_size=2, stride=2)
            else:
                self.dowmsample=nn.AvgPool2d(kernel_size=2,stride=2)

    def forward(self, input):
        return self.dowmsample(input)

# changge the h and w , and half the channels
class  Upsample(nn.Module):
    def __init__(self,inchannels,outchannels,transconv=False):
        super(Upsample, self).__init__()
        self.transconv=transconv
        if self.transconv:
            self.upsample=torch.nn.ConvTranspose2d(in_channels=inchannels,
                                     out_channels=outchannels,
                                     kernel_size=2,
                                     stride=2)
        else:
            self.upsample=nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(outchannels),
                    nn.ReLU(inplace=True))
    def forward(self, input):
        return self.upsample(input)


class MultiResUnet(nn.Module):
    '''network'''
    def __init__(self,im_ch=3,out_ch=1,alpha=1.67):
        super(MultiResUnet, self).__init__()
        self.alpha=alpha
        self.nb_filter = [64, 128, 256,512,1024]
        #self.nb_filter = [32,64, 128, 256,512]
        self.downsample=nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1=MultiResBlock(inchannels=im_ch,stage_channel=self.nb_filter[0],alpha=alpha)
        outchannel1=self.conv1.true_channels

        self.respath1=RespathConv(inchannels=outchannel1,outchannels=self.nb_filter[0],length=4)

        self.conv2=MultiResBlock(inchannels=outchannel1,stage_channel=self.nb_filter[1],alpha=alpha)
        outchannel2=self.conv2.true_channels
        self.respath2=RespathConv(inchannels=outchannel2,outchannels=self.nb_filter[1],length=3)

        self.conv3=MultiResBlock(inchannels=outchannel2,stage_channel=self.nb_filter[2],alpha=alpha)
        outchannel3=self.conv3.true_channels
        self.respath3=RespathConv(inchannels=outchannel3,outchannels=self.nb_filter[2],length=2)

        self.conv4=MultiResBlock(inchannels=outchannel3,stage_channel=self.nb_filter[3],alpha=alpha)
        outchannel4=self.conv4.true_channels
        self.respath4=RespathConv(inchannels=outchannel4,outchannels=self.nb_filter[3],length=1)

        self.conv5=MultiResBlock(inchannels=outchannel4,stage_channel=self.nb_filter[4],alpha=alpha)
        outchannel5=self.conv5.true_channels
        self.upsample1=Upsample(inchannels=outchannel5,outchannels=self.nb_filter[3])

        self.up_conv1=MultiResBlock(inchannels=self.nb_filter[3]*2,stage_channel=self.nb_filter[3])
        up_outchannel1=self.up_conv1.true_channels
        self.upsample2 = Upsample(inchannels=up_outchannel1, outchannels=self.nb_filter[2])
        self.up_conv2 = MultiResBlock(inchannels=self.nb_filter[2] * 2, stage_channel=self.nb_filter[2])
        up_outchannel2=self.up_conv2.true_channels
        self.upsample3 = Upsample(inchannels=up_outchannel2, outchannels=self.nb_filter[1])
        self.up_conv3 = MultiResBlock(inchannels=self.nb_filter[1] * 2, stage_channel=self.nb_filter[1])
        up_outchannel3=self.up_conv3.true_channels
        self.upsample4 = Upsample(inchannels=up_outchannel3, outchannels=self.nb_filter[0])
        self.up_conv4 = MultiResBlock(inchannels=self.nb_filter[0] * 2, stage_channel=self.nb_filter[0])
        up_outchannel4=self.up_conv4.true_channels
        self.final = nn.Conv2d(up_outchannel4,out_ch, kernel_size=1,stride=1,padding=0)


    def forward(self, input):

        self.conv1_o=self.conv1(input)
        self.respath1_o=self.respath1(self.conv1_o)

        self.conv2_o=self.downsample(self.conv1_o)
        self.conv2_o=self.conv2(self.conv2_o)
        self.respath2_o=self.respath2(self.conv2_o)

        self.conv3_o=self.downsample(self.conv2_o)
        self.conv3_o=self.conv3(self.conv3_o)
        self.respath3_o=self.respath3(self.conv3_o)

        self.conv4_o = self.downsample(self.conv3_o)
        self.conv4_o = self.conv4(self.conv4_o)
        self.respath4_o = self.respath4(self.conv4_o)

        self.conv5_o=self.downsample(self.conv4_o)
        self.conv5_o=self.conv5(self.conv5_o)

        self.up_conv1_o=self.upsample1(self.conv5_o)
        self.up_conv1_o=torch.cat((self.respath4_o,self.up_conv1_o),dim=1)
        self.up_conv1_o=self.up_conv1(self.up_conv1_o)

        self.up_conv2_o=self.upsample2(self.up_conv1_o)
        self.up_conv2_o=torch.cat((self.respath3_o,self.up_conv2_o),dim=1)
        self.up_conv2_o=self.up_conv2(self.up_conv2_o)

        self.up_conv3_o=self.upsample3(self.up_conv2_o)
        self.up_conv3_o=torch.cat((self.respath2_o,self.up_conv3_o),dim=1)
        self.up_conv3_o=self.up_conv3(self.up_conv3_o)

        self.up_conv4_o=self.upsample4(self.up_conv3_o)
        self.up_conv4_o=torch.cat((self.respath1_o,self.up_conv4_o),dim=1)
        self.up_conv4_o=self.up_conv4(self.up_conv4_o)

        self.final_o=self.final(self.up_conv4_o)

        return self.final_o



def calc_parameters_count(model):
    import numpy as np
    return np.sum(np.prod(v.size()) for v in model.parameters())/1e6


if __name__=="__main__":
    model=MultiResUnet()
    print(calc_parameters_count(model))



















