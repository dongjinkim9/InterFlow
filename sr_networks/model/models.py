from torch import nn
from model.common import *

import torch
import torch.nn as nn
from torch.nn import functional as F
from math import sqrt
import model.common as common
from model.registry import *


"""
    import from offical RCAN repo
    https://github.com/yulunzhang/RCAN
"""

import torch.nn as nn

def make_model(args, parent=False):
    return RCAN(args)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Channel Attention Network (RCAN)
@MODEL_REGISTRY.register()
class RCAN(nn.Module):
    def __init__(self, scale=4, conv=common.default_conv):
        super(RCAN, self).__init__()
        
        n_resgroups = 10
        n_resblocks = 20
        n_feats = 64
        kernel_size = 3
        reduction = 16 
        self.scale = scale
        act = nn.ReLU(True)
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        # caution for range of input image value
        self.sub_mean = common.MeanShift(1, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [conv(3*scale*scale, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)]

        self.add_mean = common.MeanShift(3, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

        # space to depth 
        self.space2depth = torch.nn.functional.pixel_unshuffle

    def forward(self, x):
        x = self.sub_mean(x)
        
        # added
        x = self.space2depth(x, self.scale)

        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        return self.relu(self.conv(x))


@MODEL_REGISTRY.register()
class VDSR(nn.Module):
    '''
    [*] VDSR from https://github.com/twtygqyy/pytorch-vdsr/blob/master/vdsr.py
    '''
    def __init__(self, rgb=True):
        super(VDSR, self).__init__()
        in_out_channel = 3 if(rgb) else 1
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=in_out_channel, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=in_out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)


    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        return out


def make_model(args, parent=False):
    return KPN(args)


class basicblock(nn.Module):
    def __init__(self,in_feats,out_feats,ksize=3):
        super(basicblock,self).__init__()
        block = [
            nn.Conv2d(in_feats, out_feats, kernel_size=ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_feats, out_feats, kernel_size=ksize, padding=1),        
        ]
        self.block = nn.Sequential(*block)
    def forward(self, x):
        return self.block(x)+x


class backbone(nn.Sequential):
    def __init__(self,in_feats,out_feats,n_blocks=16, ksize=3):
        m = [nn.Conv2d(in_feats,out_feats,kernel_size=ksize, padding=1)]
        for _ in range(n_blocks):
            m.append(basicblock(out_feats,out_feats,ksize))
        m.append(nn.Conv2d(out_feats,out_feats,kernel_size=ksize, padding=1))
        super(backbone, self).__init__(*m)

@MODEL_REGISTRY.register()
class KPN(nn.Module):
    """
        unoffical model from
        https://github.com/Alan-xw/RealSR
    """
    def __init__(self):
        super(KPN,self).__init__()
        n_colors = 3
        n_feats = 64

        self.shuffle_up_4= nn.PixelShuffle(4)
        self.shuffle_up_2= nn.PixelShuffle(2)
        '''  
        we use "self.head" block for processing RGB image input(3xHxW).
        In KPN, they only process Y channel of YCrCb Image input(1xHxW).
        If you hope to do as the KPN,You can replace self.head and self.backbone with:
        self.head = Shuffle_d(scale = 4)
        self.backbone = backbone(4*4, n_feats)
        '''
        self.head = nn.Sequential(*[
            nn.Conv2d(n_colors,n_feats//4,3,padding=1,stride=1),
            nn.Conv2d(n_feats//4,n_feats//16,3,padding=1,stride=1),
            Shuffle_d(scale = 4)
        ])
        self.backbone = backbone(n_feats, n_feats)
        self.laplacian = Laplacian_pyramid()
        self.laplacian_rec = Laplacian_reconstruction()
        # Branches
        self.branch_1 = pixelConv(in_feats=n_feats//16, out_feats=3, rate=4, ksize=5) 
        self.branch_2 = pixelConv(in_feats=n_feats//4, out_feats=3, rate=2, ksize=5) 
        self.branch_3 = pixelConv(in_feats=n_feats, out_feats=3, rate=1, ksize=5) 

    def forward(self, x):
          
        # Lapcian Pyramid

        Gaussian_lists,Laplacian_lists = self.laplacian(x)
        Lap_1,Lap_2 = Laplacian_lists[0],Laplacian_lists[1]
        Gaussian_3 = Gaussian_lists[-1]
        
        f_ = self.head(x)
        f_ = self.backbone(f_)  
        # branch_1
        f_1 = self.shuffle_up_4(f_)
        out_1 = self.branch_1(f_1,Lap_1)
        # branch_2
        f_2 = self.shuffle_up_2(f_)
        out_2 = self.branch_2(f_2, Lap_2)
        # branch_3
        out_3 = self.branch_3(f_, Gaussian_3)        
        
        # Laplacian Reconstruction
        merge_x2 = self.laplacian_rec(out_2,out_3)
        merge = self.laplacian_rec(out_1,merge_x2)
        
        return merge


# XuecaiHu / Meta-SR-Pytorch
# https://github.com/XuecaiHu/Meta-SR-Pytorch/blob/0.4.0/model/metardn.py

# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

# from model import common
# import time
# import torch
# import torch.nn as nn
# import math

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=( kSize -1 ) //2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c* G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class Pos2Weight(nn.Module):
    def __init__(self,inC, kernel_size=3, outC=3):
        super(Pos2Weight,self).__init__()
        self.inC = inC
        self.kernel_size=kernel_size
        self.outC = outC
        self.meta_block=nn.Sequential(
            nn.Linear(3,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,self.kernel_size*self.kernel_size*self.inC*self.outC)
        )
    def forward(self,x):

        output = self.meta_block(x)
        return output

@MODEL_REGISTRY.register()
class MetaRDN(nn.Module):
    def __init__(
        self, 
        scale: float = 1.0, 
        G0: int = 64, 
        RDNkSize: int = 3,
        RDNconfig: str = 'B',
        rgb_range: int = 255,
        n_colors: int = 3,):
        """
        args:
        scale: 'super resolution scale'
        G0: 'default number of filters. (Use in RDN)'
        RDNkSize: 'default kernel size. (Use in RDN)'
        RDNconfig: 'parameters config of RDN. (Use in RDN)'
        rgb_range: 'maximum value of RGB'
        n_colors: 'number of color channels to use'
        """
        super(MetaRDN, self).__init__()
        G0 = G0
        kSize = RDNkSize
        self.scale = scale
        # self.scale_idx = 0
        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[RDNconfig]

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )
        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        ## position to weight
        self.P2W = Pos2Weight(inC=G0)

    def repeat_x(self,x):
        scale_int = math.ceil(self.scale)
        N,C,H,W = x.size()
        x = x.view(N,C,H,1,W,1)

        x = torch.cat([x]*scale_int,3)
        x = torch.cat([x]*scale_int,5).permute(0,3,5,1,2,4)

        return x.contiguous().view(-1, C, H, W)

    def set_scale(self, scale:float):
        self.scale = scale

    # original
    def forward(self, x, pos_mat):
        #d1 =time.time()
        x = self.sub_mean(x)
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1),-1))   ###   (outH*outW, outC*inC*kernel_size*kernel_size)
        #print(d2)
        up_x = self.repeat_x(x)     ### the output is (N*r*r,inC,inH,inW)

        cols = nn.functional.unfold(up_x, 3,padding=1)
        scale_int = math.ceil(self.scale)

        cols = cols.contiguous().view(cols.size(0)//(scale_int**2),scale_int**2, cols.size(1), cols.size(2), 1).permute(0,1, 3, 4, 2).contiguous()

        local_weight = local_weight.contiguous().view(x.size(2),scale_int, x.size(3),scale_int,-1,3).permute(1,3,0,2,4,5).contiguous()
        local_weight = local_weight.contiguous().view(scale_int**2, x.size(2)*x.size(3),-1, 3)

        out = torch.matmul(cols,local_weight).permute(0,1,4,2,3)
        out = out.contiguous().view(x.size(0),scale_int,scale_int,3,x.size(2),x.size(3)).permute(0,3,4,1,5,2)
        out = out.contiguous().view(x.size(0),3, scale_int*x.size(2),scale_int*x.size(3))
        out = self.add_mean(out)

        return out

    def set_scale(self, scale:float):
        self.scale = scale

    # def set_scale(self, scale_idx):
    #     self.scale_idx = scale_idx
    #     self.scale = self.args.scale[scale_idx]

    ######by given the scale and the size of input image
    ######we caculate the input matrix for the weight prediction network
    ###### input matrix for weight prediction network
    def input_matrix_wpn(self,inH, inW, scale, add_scale=True):
        '''
        inH, inW: the size of the feature maps
        scale: is the upsampling times
        '''
        outH, outW = int(scale*inH), int(scale*inW)

        #### mask records which pixel is invalid, 1 valid or o invalid
        #### h_offset and w_offset caculate the offset to generate the input matrix
        scale_int = int(math.ceil(scale))
        h_offset = torch.ones(inH, scale_int, 1)
        mask_h = torch.zeros(inH,  scale_int, 1)
        w_offset = torch.ones(1, inW, scale_int)
        mask_w = torch.zeros(1, inW, scale_int)
        if add_scale:
            scale_mat = torch.zeros(1,1)
            scale_mat[0,0] = 1.0/scale
            #res_scale = scale_int - scale
            #scale_mat[0,scale_int-1]=1-res_scale
            #scale_mat[0,scale_int-2]= res_scale
            scale_mat = torch.cat([scale_mat]*(inH*inW*(scale_int**2)),0)  ###(inH*inW*scale_int**2, 4)

        ####projection  coordinate  and caculate the offset 
        h_project_coord = torch.arange(0,outH, 1).float().mul(1.0/scale)
        int_h_project_coord = torch.floor(h_project_coord)

        offset_h_coord = h_project_coord - int_h_project_coord
        int_h_project_coord = int_h_project_coord.int()

        w_project_coord = torch.arange(0, outW, 1).float().mul(1.0/scale)
        int_w_project_coord = torch.floor(w_project_coord)

        offset_w_coord = w_project_coord - int_w_project_coord
        int_w_project_coord = int_w_project_coord.int()

        ####flag for   number for current coordinate LR image
        flag = 0
        number = 0
        for i in range(outH):
            if int_h_project_coord[i] == number:
                h_offset[int_h_project_coord[i], flag, 0] = offset_h_coord[i]
                mask_h[int_h_project_coord[i], flag,  0] = 1
                flag += 1
            else:
                h_offset[int_h_project_coord[i], 0, 0] = offset_h_coord[i]
                mask_h[int_h_project_coord[i], 0, 0] = 1
                number += 1
                flag = 1

        flag = 0
        number = 0
        for i in range(outW):
            if int_w_project_coord[i] == number:
                w_offset[0, int_w_project_coord[i], flag] = offset_w_coord[i]
                mask_w[0, int_w_project_coord[i], flag] = 1
                flag += 1
            else:
                w_offset[0, int_w_project_coord[i], 0] = offset_w_coord[i]
                mask_w[0, int_w_project_coord[i], 0] = 1
                number += 1
                flag = 1

        ## the size is scale_int* inH* (scal_int*inW)
        h_offset_coord = torch.cat([h_offset] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
        w_offset_coord = torch.cat([w_offset] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)
        ####
        mask_h = torch.cat([mask_h] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
        mask_w = torch.cat([mask_w] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)

        pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
        mask_mat = torch.sum(torch.cat((mask_h,mask_w),2),2).view(scale_int*inH,scale_int*inW)
        mask_mat = mask_mat.eq(2)
        pos_mat = pos_mat.contiguous().view(1, -1,2)
        if add_scale:
            pos_mat = torch.cat((scale_mat.view(1,-1,1), pos_mat),2)

        return pos_mat,mask_mat ##outH*outW*2 outH=scale_int*inH , outW = scale_int *inW

# https://github.com/sanghyun-son/EDSR-PyTorch/blob/master/src/demo.sh
# EDSR in the paper (x4) - from EDSR (x2)
# python main.py --model EDSR --scale 4 --save edsr_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR_x2 model dir]

@MODEL_REGISTRY.register()
class MetaEDSR(nn.Module):
    def __init__(
        self, 
        conv=common.default_conv,
        # EDSR-baseline
        n_resblocks: int = 16,
        n_feats: int = 64,
        res_scale : float = 1.0,
        # EDSR
        # n_resblocks: int = 32,
        # n_feats: int = 256,
        # res_scale : float = 0.1,
        scale : float = 1.0,
        rgb_range : int = 255,
        n_colors : int = 3,
        ):
        """
        args:
        scale: 'super resolution scale'
        G0: 'default number of filters. (Use in RDN)'
        RDNkSize: 'default kernel size. (Use in RDN)'
        RDNconfig: 'parameters config of RDN. (Use in RDN)'
        rgb_range: 'maximum value of RGB'
        n_colors: 'number of color channels to use'
        """
        
        super(MetaEDSR, self).__init__()
        # raise NotImplementedError()
        n_resblock = n_resblocks
        n_feats = n_feats
        kernel_size = 3
        scale = scale
        act = nn.ReLU(True)

        self.scale_idx = 0
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))



        self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        ## position to weight

        self.P2W = Pos2Weight(inC=n_feats)

    def repeat_x(self,x):
        scale_int = math.ceil(self.scale)
        N,C,H,W = x.size()
        x = x.view(N,C,H,1,W,1)

        x = torch.cat([x]*scale_int,3)
        x = torch.cat([x]*scale_int,5).permute(0,3,5,1,2,4)

        return x.contiguous().view(-1, C, H, W)

    def forward(self, x, pos_mat):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        local_weight = self.P2W(pos_mat.view(pos_mat.size(1),-1))   ###   (outH*outW, outC*inC*kernel_size*kernel_size)
        up_x = self.repeat_x(res)     ### the output is (N*r*r,inC,inH,inW)

        # N*r^2 x [inC * kH * kW] x [inH * inW]
        cols = nn.functional.unfold(up_x, 3,padding=1)
        scale_int = math.ceil(self.scale)

        cols = cols.contiguous().view(cols.size(0)//(scale_int**2),scale_int**2, cols.size(1), cols.size(2), 1).permute(0,1, 3, 4, 2).contiguous()

        local_weight = local_weight.contiguous().view(x.size(2),scale_int, x.size(3),scale_int,-1,3).permute(1,3,0,2,4,5).contiguous()
        local_weight = local_weight.contiguous().view(scale_int**2, x.size(2)*x.size(3),-1, 3)

        out = torch.matmul(cols,local_weight).permute(0,1,4,2,3)
        out = out.contiguous().view(x.size(0),scale_int,scale_int,3,x.size(2),x.size(3)).permute(0,3,4,1,5,2)
        out = out.contiguous().view(x.size(0),3, scale_int*x.size(2),scale_int*x.size(3))
        out = self.add_mean(out)
        ###
        return out
    
    ######by given the scale and the size of input image
    ######we caculate the input matrix for the weight prediction network
    ###### input matrix for weight prediction network
    def input_matrix_wpn(self,inH, inW, scale, add_scale=True):
        '''
        inH, inW: the size of the feature maps
        scale: is the upsampling times
        '''
        outH, outW = int(scale*inH), int(scale*inW)

        #### mask records which pixel is invalid, 1 valid or o invalid
        #### h_offset and w_offset caculate the offset to generate the input matrix
        scale_int = int(math.ceil(scale))
        h_offset = torch.ones(inH, scale_int, 1)
        mask_h = torch.zeros(inH,  scale_int, 1)
        w_offset = torch.ones(1, inW, scale_int)
        mask_w = torch.zeros(1, inW, scale_int)
        if add_scale:
            scale_mat = torch.zeros(1,1)
            scale_mat[0,0] = 1.0/scale
            #res_scale = scale_int - scale
            #scale_mat[0,scale_int-1]=1-res_scale
            #scale_mat[0,scale_int-2]= res_scale
            scale_mat = torch.cat([scale_mat]*(inH*inW*(scale_int**2)),0)  ###(inH*inW*scale_int**2, 4)

        ####projection  coordinate  and caculate the offset 
        h_project_coord = torch.arange(0,outH, 1).float().mul(1.0/scale)
        int_h_project_coord = torch.floor(h_project_coord)

        offset_h_coord = h_project_coord - int_h_project_coord
        int_h_project_coord = int_h_project_coord.int()

        w_project_coord = torch.arange(0, outW, 1).float().mul(1.0/scale)
        int_w_project_coord = torch.floor(w_project_coord)

        offset_w_coord = w_project_coord - int_w_project_coord
        int_w_project_coord = int_w_project_coord.int()

        ####flag for   number for current coordinate LR image
        flag = 0
        number = 0
        for i in range(outH):
            if int_h_project_coord[i] == number:
                h_offset[int_h_project_coord[i], flag, 0] = offset_h_coord[i]
                mask_h[int_h_project_coord[i], flag,  0] = 1
                flag += 1
            else:
                h_offset[int_h_project_coord[i], 0, 0] = offset_h_coord[i]
                mask_h[int_h_project_coord[i], 0, 0] = 1
                number += 1
                flag = 1

        flag = 0
        number = 0
        for i in range(outW):
            if int_w_project_coord[i] == number:
                w_offset[0, int_w_project_coord[i], flag] = offset_w_coord[i]
                mask_w[0, int_w_project_coord[i], flag] = 1
                flag += 1
            else:
                w_offset[0, int_w_project_coord[i], 0] = offset_w_coord[i]
                mask_w[0, int_w_project_coord[i], 0] = 1
                number += 1
                flag = 1

        ## the size is scale_int* inH* (scal_int*inW)
        h_offset_coord = torch.cat([h_offset] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
        w_offset_coord = torch.cat([w_offset] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)
        ####
        mask_h = torch.cat([mask_h] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
        mask_w = torch.cat([mask_w] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)

        pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
        mask_mat = torch.sum(torch.cat((mask_h,mask_w),2),2).view(scale_int*inH,scale_int*inW)
        mask_mat = mask_mat.eq(2)
        pos_mat = pos_mat.contiguous().view(1, -1,2)
        if add_scale:
            pos_mat = torch.cat((scale_mat.view(1,-1,1), pos_mat),2)

        return pos_mat,mask_mat ##outH*outW*2 outH=scale_int*inH , outW = scale_int *inW

    def set_scale(self, scale:float):
        self.scale = scale


# from model import common
# import torch
# import torch.nn as nn
# import pdb

"""
HAN 
https://github.com/wwlCape/HAN/blob/master/src/model/han.py
"""

def make_model(args, parent=False):
    return HAN(args)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class LAM_Module(nn.Module):
    """ Layer attention module"""
    def __init__(self, in_dim):
        super(LAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()
        proj_query = x.view(m_batchsize, N, -1)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, N, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out + x
        out = out.view(m_batchsize, -1, height, width)
        return out

class CSAM_Module(nn.Module):
    """ Channel-Spatial attention module"""
    def __init__(self, in_dim):
        super(CSAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        #self.softmax  = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))
        
        # proj_query = x.view(m_batchsize, N, -1)
        # proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        # energy = torch.bmm(proj_query, proj_key)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        # attention = self.softmax(energy_new)
        # proj_value = x.view(m_batchsize, N, -1)

        # out = torch.bmm(attention, proj_value)
        # out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out
        out = out.view(m_batchsize, -1, height, width)
        x = x * out + x
        return x

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res



## Holistic Attention Network (HAN)
@MODEL_REGISTRY.register()
class HAN(nn.Module):
    def __init__(self, conv=common.default_conv, scale=4, rgb_range: int = 1):
        super(HAN, self).__init__()
        
        # import default settings from
        # https://github.com/wwlCape/HAN/blob/master/src/template.py
        
        self.scale = scale
        n_resgroups = 10
        n_resblocks = 20
        n_feats = 64
        kernel_size = 3
        reduction = 16
        res_scale = 1
        act = nn.ReLU(True)
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [conv(3*4*4, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)]

        self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.csa = CSAM_Module(n_feats)
        self.la = LAM_Module(n_feats)
        self.last_conv = nn.Conv2d(n_feats*11, n_feats, 3, 1, 1)
        self.last = nn.Conv2d(n_feats*2, n_feats, 3, 1, 1)
        self.tail = nn.Sequential(*modules_tail)

        # space to depth 
        self.space2depth = torch.nn.functional.pixel_unshuffle

    def forward(self, x):
        x = self.sub_mean(x)
        
        # added
        x = self.space2depth(x, self.scale)

        x = self.head(x)
        res = x
        #pdb.set_trace()
        for name, midlayer in self.body._modules.items():
            res = midlayer(res)
            #print(name)
            if name=='0':
                res1 = res.unsqueeze(1)
            else:
                res1 = torch.cat([res.unsqueeze(1),res1],1)
        #res = self.body(x)
        out1 = res
        #res3 = res.unsqueeze(1)
        #res = torch.cat([res1,res3],1)
        res = self.la(res1)
        out2 = self.last_conv(res)

        out1 = self.csa(out1)
        out = torch.cat([out1, out2], 1)
        res = self.last(out)
        
        res += x
        #res = self.csa(res)

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


"""
https://github.com/HarukiYqM/Non-Local-Sparse-Attention/blob/main/src/model/nlsn.py
"""
import model.nlsn.attention as attention
import model.nlsn.nlsn_common as nlsn_common

def make_model(args, parent=False):
    if args.dilation:
        from model import dilated
        return NLSN(args, dilated.dilated_conv)
    else:
        return NLSN(args)

@MODEL_REGISTRY.register()
class NLSN(nn.Module):
    def __init__(self, conv=nlsn_common.default_conv, scale = 4, rgb_range: int = 1):
        super(NLSN, self).__init__()
        
        # import default settings from
        # https://github.com/HarukiYqM/Non-Local-Sparse-Attention/blob/main/src/demo.sh

        n_resblock = 32
        n_feats = 256
        kernel_size = 3 
        self.scale = scale
        chunk_size = 144
        n_hashes = 4
        res_scale = 0.1
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = nlsn_common.MeanShift(rgb_range, rgb_mean, rgb_std)
        m_head = [conv(3*4*4, n_feats, kernel_size)]

        # define body module
        m_body = [attention.NonLocalSparseAttention(
            channels=n_feats, chunk_size=chunk_size, n_hashes=n_hashes, reduction=4, res_scale=res_scale)]         

        for i in range(n_resblock):
            m_body.append( nlsn_common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ))
            if (i+1)%8==0:
                m_body.append(attention.NonLocalSparseAttention(
                    channels=n_feats, chunk_size=chunk_size, n_hashes=n_hashes, reduction=4, res_scale=res_scale))
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            nlsn_common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(
                n_feats, 3, kernel_size,
                padding=(kernel_size//2)
            )
        ]

        self.add_mean = nlsn_common.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        # space to depth 
        self.space2depth = torch.nn.functional.pixel_unshuffle

    def forward(self, x):
        x = self.sub_mean(x)

        # added
        x = self.space2depth(x, self.scale)

        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

"""
https://github.com/dvlab-research/Simple-SR/blob/master/exps/LAPAR_A_x2/network.py
"""

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.LAPAR_A.lightWeightNet import WeightNet


class ComponentDecConv(nn.Module):
    def __init__(self, k_path, k_size):
        super(ComponentDecConv, self).__init__()

        kernel = pickle.load(open(k_path, 'rb'))
        kernel = torch.from_numpy(kernel).float().view(-1, 1, k_size, k_size)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        out = F.conv2d(x, weight=self.weight, bias=None, stride=1, padding=0, groups=1)

        return out

@MODEL_REGISTRY.register()
class LAPAR_A(nn.Module):
    def __init__(self, scale = 4, kernel_path=""):
        super(LAPAR_A, self).__init__()

        self.k_size = 5
        # MODEL.SCALE = DATASET.SCALE
        self.s = scale
        in_chl = 3*4*4
        nf = 32
        n_block = 4
        out_chl = 72
        scale = 4

        self.w_conv = WeightNet(in_chl, nf, n_block, out_chl, scale)
        self.decom_conv = ComponentDecConv(kernel_path, self.k_size)

        self.criterion = nn.L1Loss(reduction='mean')


    def forward(self, x, gt=None):
        B, C, H, W = x.size()

        bic = F.interpolate(x, scale_factor=self.s, mode='bicubic', align_corners=False)
        pad = self.k_size // 2
        x_pad = F.pad(bic, pad=(pad, pad, pad, pad), mode='reflect')
        pad_H, pad_W = x_pad.size()[2:]
        x_pad = x_pad.view(B * 3, 1, pad_H, pad_W)
        x_com = self.decom_conv(x_pad).view(B, 3, -1, self.s * H, self.s * W)  # B, 3, N_K, Hs, Ws

        weight = self.w_conv(x)
        weight = weight.view(B, 1, -1, self.s * H, self.s * W)  # B, 1, N_K, Hs, Ws

        out = torch.sum(weight * x_com, dim=2)

        if gt is not None:
            loss_dict = dict(L1=self.criterion(out, gt))
            return loss_dict
        else:
            return out



###
# https://github.com/yinboc/liif
##

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class EDSR(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(EDSR, self).__init__()
        url = {
            'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
            'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
            'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
            'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
            'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
            'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
        }
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        if args.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = args.n_colors
            # define tail module
            m_tail = [
                Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ]
            self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        #x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        if self.args.no_upsampling:
            x = res
        else:
            x = self.tail(res)
        #x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

from argparse import Namespace

def make_edsr_baseline(n_resblocks=16, n_feats=64, res_scale=1,
                       scale=2, no_upsampling=False, rgb_range=1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    args.n_colors = 3
    return EDSR(args)

class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        self.args = args
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        if args.no_upsampling:
            self.out_dim = G0
        else:
            self.out_dim = args.n_colors
            # Up-sampling net
            if r == 2 or r == 3:
                self.UPNet = nn.Sequential(*[
                    nn.Conv2d(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(r),
                    nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
                ])
            elif r == 4:
                self.UPNet = nn.Sequential(*[
                    nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
                ])
            else:
                raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1

        if self.args.no_upsampling:
            return x
        else:
            return self.UPNet(x)


def make_rdn(G0=64, RDNkSize=3, RDNconfig='B',
             scale=2, no_upsampling=False):
    args = Namespace()
    args.G0 = G0
    args.RDNkSize = RDNkSize
    args.RDNconfig = RDNconfig

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.n_colors = 3
    return RDN(args)


@MODEL_REGISTRY.register()
class LIIF(nn.Module):

    def __init__(self, encoder_type= 'EDSR_baseline', imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True,
                 no_upsampling=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        # self.encoder = models.make(encoder_spec)
        if encoder_type == 'EDSR_baseline':
            self.encoder = make_edsr_baseline(no_upsampling=no_upsampling)
        elif encoder_type == 'RDN':
            self.encoder = make_rdn(no_upsampling=no_upsampling)
        else:
            raise NotImplementedError()

        imnet_spec = {
            'name': 'mlp',
            'args': {
                'out_dim': 3,
                'hidden_list': [256, 256, 256, 256]
            }
        }

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                imnet_in_dim *= 9
            imnet_in_dim += 2 # attach coord
            if self.cell_decode:
                imnet_in_dim += 2
            # self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
            self.imnet = MLP(in_dim=imnet_in_dim, out_dim=3, hidden_list=[256, 256, 256, 256])
        else:
            self.imnet = None

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)


@MODEL_REGISTRY.register()        
class MetaSR(nn.Module):

    def __init__(self, encoder_type= 'EDSR_baseline', no_upsampling=True):
        super().__init__()

        if encoder_type == 'EDSR_baseline':
            self.encoder = make_edsr_baseline(no_upsampling=no_upsampling)
        elif encoder_type == 'RDN':
            self.encoder = make_rdn(no_upsampling=no_upsampling)
        else:
            raise NotImplementedError()
        
        self.imnet = MLP(
            in_dim = 3,
            out_dim = self.encoder.out_dim * 9 * 3,
            hidden_list = [256])

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        feat = F.unfold(feat, 3, padding=1).view(
            feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda()
        feat_coord[:, :, 0] -= (2 / feat.shape[-2]) / 2
        feat_coord[:, :, 1] -= (2 / feat.shape[-1]) / 2
        feat_coord = feat_coord.permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        coord_ = coord.clone()
        coord_[:, :, 0] -= cell[:, :, 0] / 2
        coord_[:, :, 1] -= cell[:, :, 1] / 2
        coord_q = (coord_ + 1e-6).clamp(-1 + 1e-6, 1 - 1e-6)
        q_feat = F.grid_sample(
            feat, coord_q.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_coord = F.grid_sample(
            feat_coord, coord_q.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)

        rel_coord = coord_ - q_coord
        rel_coord[:, :, 0] *= feat.shape[-2] / 2
        rel_coord[:, :, 1] *= feat.shape[-1] / 2

        r_rev = cell[:, :, 0] * (feat.shape[-2] / 2)
        inp = torch.cat([rel_coord, r_rev.unsqueeze(-1)], dim=-1)

        bs, q = coord.shape[:2]
        pred = self.imnet(inp.view(bs * q, -1)).view(bs * q, feat.shape[1], 3)
        pred = torch.bmm(q_feat.contiguous().view(bs * q, 1, -1), pred)
        pred = pred.view(bs, q, 3)
        return pred

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)

