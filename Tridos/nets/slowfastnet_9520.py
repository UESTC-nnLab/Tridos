import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
from einops.layers.torch import Rearrange
from .module.video_swin import SwinTransformerBlock3D
from .module.STDM import TDM_S
from .module.dtum import  Res_CBAM_block

class YOLOPAFPN(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0, in_features = ("dark3", "dark4", "dark5"), in_channels = [256, 512, 1024], depthwise = False, act = "silu"):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act)
        self.in_features    = in_features

        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        self.lateral_conv0  = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
    
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )  

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        self.reduce_conv1   = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )
        

        # self.bu_conv2       = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        # self.C3_n3 = CSPLayer(
        #     int(2 * in_channels[0] * width),
        #     int(in_channels[1] * width),
        #     round(3 * depth),
        #     False,
        #     depthwise = depthwise,
        #     act = act,
        # )
        # self.bu_conv1       = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        # self.C3_n4 = CSPLayer(
        #     int(2 * in_channels[1] * width),
        #     int(in_channels[2] * width),
        #     round(3 * depth),
        #     False,
        #     depthwise = depthwise,
        #     act = act,
        # )


    def forward(self, input):
        out_features            = self.backbone.forward(input)
        [feat1, feat2, feat3]   = [out_features[f] for f in self.in_features]

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        P5          = self.lateral_conv0(feat3)
        #-------------------------------------------#
        #  20, 20, 512 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.upsample(P5)
        #-------------------------------------------#
        #  40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
        #-------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.C3_p4(P5_upsample)

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        P4          = self.reduce_conv1(P5_upsample) 
        #-------------------------------------------#
        #   40, 40, 256 -> 80, 80, 256
        #-------------------------------------------#
        P4_upsample = self.upsample(P4) 
        #-------------------------------------------#
        #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
        #-------------------------------------------#
        P4_upsample = torch.cat([P4_upsample, feat1], 1) 
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        P3_out      = self.C3_p3(P4_upsample)  

        
        # P3_downsample   = self.bu_conv2(P3_out) 
        # P3_downsample   = torch.cat([P3_downsample, P4], 1) 
        # P4_out          = self.C3_n3(P3_downsample) 
        # P4_downsample   = self.bu_conv1(P4_out)
        # P4_downsample   = torch.cat([P4_downsample, P5], 1)
        # P5_out          = self.C3_n4(P4_downsample)
        # return (P3_out, P4_out, P5_out)
        return P3_out

class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [16, 32, 64], act = "silu"):
        super().__init__()
        Conv            =  BaseConv
        
        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )
            

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        #---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        #---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            #---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            #---------------------------------------------------#
            x       = self.stems[k](x)
            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            cls_feat    = self.cls_convs[k](x)
            #---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            #---------------------------------------------------#
            cls_output  = self.cls_preds[k](cls_feat)

            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            reg_feat    = self.reg_convs[k](x)
            #---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            #---------------------------------------------------#
            reg_output  = self.reg_preds[k](reg_feat)
            #---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            #---------------------------------------------------#
            obj_output  = self.obj_preds[k](reg_feat)

            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs

class Neck(nn.Module):
    def __init__(self, channels=[128,256,512] ,num_frame=5):
        super().__init__()
        self.num_frame = num_frame

        #  #关键帧与参考帧融合
        self.conv_ref = nn.Sequential(
            BaseConv(channels[0]*(self.num_frame-1), channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1, act='sigmoid')
        )
        self.conv_cur = BaseConv(channels[0], channels[0],3,1)
        # 最终融合
        self.conv_cr_mix = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )

        self.conv_fin_mix = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1),
        )
        self.conv_fre_mix = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1),
        )
        self.nolocal = NonLocalBlock(128)
    
        self.swin = SwinTransformerBlock3D(128,num_frames=self.num_frame)
        self.conv_t = nn.Sequential(
            BaseConv(channels[0]*(self.num_frame), channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )

        self.agpf = AGPF(128)
        self.conv_fre = nn.Sequential(
            BaseConv(channels[0]*5, channels[0]*5,3,1),
            BaseConv(channels[0]*5,channels[0],3,1),
        )


        ####memory####
        self.keyvalue_Q = KeyValue_Q(128,64,128)
        self.keyvalue_M = KeyValue_M(128,64,128)
        self.memory = MemoryReader()
        self.resblock0 = Res_CBAM_block(in_channels=channels[0]*2,out_channels=channels[0])
        self.resblock1 = Res_CBAM_block(in_channels=channels[0]*2,out_channels=channels[0])
        self.resblock2 = Res_CBAM_block(in_channels=channels[0]*2,out_channels=channels[0])
        self.tdms = TDM_S(nframes=self.num_frame)

        

    def forward(self, feats):
        f_feats = []   # 5* 4,128,64,64

        rc_feat = torch.cat([feats[j] for j in range(self.num_frame-1)],dim=1)  
        r_feat = self.conv_ref(rc_feat)  
        c_feat = self.conv_cur(r_feat*feats[-1]) 
        c_feat = self.conv_cr_mix(torch.cat([c_feat, feats[-1]], dim=1)) 

        c_feat = self.nolocal(c_feat)

        K_Q, V_Q = self.keyvalue_Q(feats[-1])
        K_M, V_M = self.keyvalue_M(c_feat)
        c_feat = self.memory(K_M, V_M, K_Q, V_Q)##空间记忆模块

        differ = self.tdms(feats) #时间差分模块

        c_feat = self.conv_fin_mix(torch.cat([c_feat,differ], dim=1))

        # #频域特征
        p_feats = []
        for i in range(self.num_frame):
            temp_f = self.agpf(feats[i])
            p_feats.append(temp_f)
        p_feat1 = torch.cat([p_feats[i] for i in range(self.num_frame)], dim=1)
        pt_feat = self.conv_fre(p_feat1)

        pc_feat = self.resblock1(torch.cat([pt_feat,c_feat], dim=1))
        
        p_feat = self.swin(p_feat1)
        p_feat = self.conv_t(p_feat)


        pc = self.resblock0(torch.cat([p_feat,c_feat], dim=1))

        
        f_feat = self.resblock2(torch.cat([pc_feat,pc], dim=1))
        
        
        f_feats.append(f_feat)
        
        return f_feats

class slowfastnet(nn.Module):
    def __init__(self, num_classes, fp16=False, num_frame=5):
        super(slowfastnet, self).__init__()
        self.num_frame = num_frame
        self.backbone = YOLOPAFPN(0.33,0.50) 

        #-----------------------------------------#
        #   尺度感知模块
        #-----------------------------------------#
        self.neck = Neck(channels=[128,256,512], num_frame=num_frame)
        #----------------------------------------------------------#
        #   head
        #----------------------------------------------------------#
        self.head = YOLOXHead(num_classes=num_classes, width = 1.0, in_channels = [128], act = "silu")

    def forward(self, inputs):  #input=[4,3,5,512,512]  B C N H W
        feat = []  
        for i in range(self.num_frame):
            feat.append(self.backbone(inputs[:,:,i,:,:]))  # feat里是每一帧的特征
        """5*[4,128,64,64] [b,256,32,32][b,512,16,16]"""
        
        if self.neck:
            feat = self.neck(feat)
        
        outputs  = self.head(feat)
      
        return  outputs    # 计算损失那边 的anchor 应该是 [1, M, 4] size的

class NonLocalBlock(nn.Module):
    def __init__(self, planes, reduce_ratio=8):
        super(NonLocalBlock, self).__init__()

        inter_planes = planes // reduce_ratio
        self.query_conv = nn.Conv2d(planes, inter_planes, kernel_size=1)
        self.key_conv = nn.Conv2d(planes, inter_planes, kernel_size=1)
        self.value_conv = nn.Conv2d(planes, planes, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()

        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        proj_query = proj_query.contiguous().view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = proj_key.contiguous().view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = proj_value.contiguous().view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, -1, height, width)

        out = self.gamma * out + x
        return out

class AGPF(nn.Module):
    def __init__(self, n_feat, n_resblocks=1):
        super(AGPF, self).__init__()
        modules_body = [
            Freq_block(n_feat) for _ in range(n_resblocks)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1))
        self.body = nn.Sequential(*modules_body)
        self.re_scale = Scale(1)

    def forward(self, x):
        res = self.body(x)
        return res + self.re_scale(x)
class Freq_block(nn.Module):
    def __init__(self, dim,dfilter_freedom=[3, 2],
                 dfilter_type='piecewise_linear'):
        super().__init__()
        self.dim = dim
        self.dw_amp_conv = nn.Sequential(
            nn.Conv2d(dim, dim, groups=dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.df1 = nn.Sequential(
            nn.Conv2d(2, 2, groups=2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.df2 = nn.Sequential(
            nn.Conv2d(2, 2, groups=2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.dw_pha_conv = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, groups=dim*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim*2, dim, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
            )

    def forward(self, x):
        b,c,h,w = x.shape
        msF = torch.fft.rfft2(x+1e-8, dim=(-2, -1))
        msF = torch.cat([
            msF[:, :, msF.size(2) // 2 + 1:, :],
            msF[:, :, :msF.size(2) // 2 + 1, :]], dim=2)
        # msF = torch.fft.fftshift(msF, dim=(-2, -1))
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)

        amp_fuse = self.dw_amp_conv(msF_amp)
        avg_attn = torch.mean(amp_fuse, dim=1, keepdim=True)
        max_attn, _ = torch.max(amp_fuse, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        agg=self.df1(agg)
        amp_fuse=amp_fuse*agg
        amp_res = amp_fuse - msF_amp
        pha_guide=torch.cat((msF_pha,amp_res),dim=1)
        pha_fuse = self.dw_pha_conv(pha_guide)
        avg_attn = torch.mean(pha_fuse, dim=1, keepdim=True)
        max_attn, _ = torch.max(pha_fuse, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        agg = self.df2(agg)
        pha_fuse = pha_fuse * agg
        pha_fuse=pha_fuse*(2.*math.pi)-math.pi
        # pha_fuse = torch.clamp(pha_fuse, -math.pi, math.pi)
        ## amp_fuse = amp_fuse + msF_amp
        # pha_fuse = pha_fuse + msF_pha

        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)
        # out=torch.fft.ifftshift(out, dim=(-2, -1))
        out = torch.cat([
            out[:, :, out.size(2) // 2 - 1:, :],
            out[:, :, :out.size(2) // 2 - 1, :]], dim=2)
        out = torch.abs(torch.fft.irfft2(out+1e-8, s=(h, w)))
        if torch.isnan(out).sum()>0:
            print('freq feature include NAN!!!!')
            # assert torch.isnan(out).sum() == 0  ##这里有问题
            out = torch.nan_to_num(out, nan=1e-5, posinf=1e-5, neginf=1e-5)
        out = out + x
        return F.relu(out)
class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))
    def forward(self, input):
        return input * self.scale

def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m
def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.
class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)


    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y

class Con1x1WithBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Con1x1WithBnRelu, self).__init__()
        self.con1x1 = nn.Conv2d(in_ch, out_ch,
                                kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(True)

    def forward(self, input):
        return self.relu(self.bn(self.con1x1(input)))
class KeyValue_Q(torch.nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue_Q, self).__init__()
        self.key_conv = torch.nn.Conv2d(indim, keydim, kernel_size=3, padding=1, stride=1)
        self.value_conv = torch.nn.Conv2d(indim, valdim, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        return self.key_conv(x), self.value_conv(x)
class KeyValue_M(torch.nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue_M, self).__init__()
        self.key_conv = torch.nn.Conv2d(indim, keydim, kernel_size=3, padding=1, stride=1)
        self.value_conv = torch.nn.Conv2d(indim, valdim, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        return self.key_conv(x), self.value_conv(x)
class MemoryReader(torch.nn.Module):
    def __init__(self):
        super(MemoryReader, self).__init__()
        self.memory_reduce = Con1x1WithBnRelu(256, 128)

    def forward(self, K_M, V_M, K_Q, V_Q):  # shape: B,C,N,H,W, Eq.2 in the paper.
        B, C_K, H, W = K_M.size()
        _, C_V, _, _ = V_M.size()

        K_M = K_M.view(B, C_K,  H * W)
        K_M = torch.transpose(K_M, 1, 2)  # 4 4096 64
        K_Q = K_Q.view(B, C_K, H * W) # 4 64 4096

        w = torch.bmm(K_M, K_Q) # 4 4096 4096
        w = w / math.sqrt(C_K)
        w = F.softmax(w, dim=1)
        V_M = V_M.view(B, C_V,  H * W)  # 4 128 64*64

        mem = torch.bmm(V_M, w)
        mem = mem.view(B, C_V, H, W) # 4 128 64*64

        E_t = torch.cat([mem, V_Q], dim=1) # 4 256 64 64


        return self.memory_reduce(E_t)


if __name__ == "__main__":
    
    # from yolo_training import YOLOLoss
    net = slowfastnet(num_classes=1, num_frame=5)
    
    # bs = 4
    # a = torch.randn(bs, 3, 5, 512, 512)
    # out = net(a)
    # for item in out:
    #     print(item.size())
        
    # yolo_loss    = YOLOLoss(num_classes=1, fp16=False, strides=[16])

    # target = torch.randn([bs, 1, 5]).cuda()
    # target = nn.Softmax()(target)
    # target = [item for item in target]

    # loss = yolo_loss(out, target)c
    # print(loss)

    # net = LFS_Head(64, 10, 6)  #4 128 64 64->4 6 32 32
    # net = FAD_Head(64) # 4 128 64 64 ->4 512 64 64
    # net = ReasoningLayer2() # 4 128 64 64->4 128 64 64ccccccccc
    a = torch.randn(4, 5, 128,64,64)
    out = net(a)  
    print(out.shape)
