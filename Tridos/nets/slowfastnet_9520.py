import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
# from darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
from einops.layers.torch import Rearrange
# from .module.GAL.gal import GAL
from .module.video_swin import SwinTransformerBlock3D
from .module.STDM import TDM_S
from .module.dtum import DTUM, Res_CBAM_block
# from .module.dyhead import DyHeadBlock
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

        # self.gal = GAL(128)
        # self.dtum = DTUM(in_channels=512,num_frames=5)
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
        # ###frequency
        # ##F3Net
        # self.fre = FAD_Head(64) #img_size
        # self.conv_fre = nn.Sequential(
        #     BaseConv(channels[2], channels[2],3,1),
        #     BaseConv(channels[2],channels[0],3,1)
        # )
        # self.conv_fre = nn.Sequential(
        #     BaseConv(channels[0]*(self.num_frame)*4, channels[0]*2*4,3,1),
        #     BaseConv(channels[0]*2*4,channels[0]*2,3,1),
        #     BaseConv(channels[0]*2,channels[0],3,1)
        # )
        # ###SEIFNet
        # self.motion = CoDEM2(128)
        # self.acff = ACFF2(128)
        # self.attention = SupervisedAttentionModule(128)
        #TSFNet
        # self.rspb = RSPB(64, 3,4,act='PReLU',n_resblocks=2,norm=None)
        self.agpf = AGPF(128)
        # self.selffuse = selfFuseBlock(128)
        self.conv_fre = nn.Sequential(
            BaseConv(channels[0]*5, channels[0]*5,3,1),
            BaseConv(channels[0]*5,channels[0],3,1),
        )

        ##Octave 
        # self.fre = Octave(in_channels=128, out_channels=128)
        # self.conv_fre = nn.Sequential(
        #     BaseConv(channels[0]*(self.num_frame), channels[0]*2,3,1),
        #     BaseConv(channels[0]*2,channels[0],3,1, act='sigmoid')
        # )

        ####memory####
        self.keyvalue_Q = KeyValue_Q(128,64,128)
        self.keyvalue_M = KeyValue_M(128,64,128)
        self.memory = MemoryReader()
        # self.fuse = fuse(128,128)
        self.resblock0 = Res_CBAM_block(in_channels=channels[0]*2,out_channels=channels[0])
        self.resblock1 = Res_CBAM_block(in_channels=channels[0]*2,out_channels=channels[0])
        self.resblock2 = Res_CBAM_block(in_channels=channels[0]*2,out_channels=channels[0])
        self.tdms = TDM_S(nframes=self.num_frame)
        # self.dyhead = DyHeadBlock(in_channels=128)

        #时间特征-参考帧分别与关键帧融合
        # for i in range(1,num_frame):
        #     self.__setattr__("attn_%d"%i, ReasoningLayer2()) 
        

    def forward(self, feats):
        f_feats = []   # 5* 4,128,64,64
        # for i in range(self.num_frame):
        #     feats[i] = self.nolocal(feats[i])
        #空间特征
        rc_feat = torch.cat([feats[j] for j in range(self.num_frame-1)],dim=1)  # 参考帧在通道维度融合， 4, 512，64,64
        r_feat = self.conv_ref(rc_feat)  #4,128，64,64  通过sigmoid计算权重
        c_feat = self.conv_cur(r_feat*feats[-1]) #和关键帧相乘
        c_feat = self.conv_cr_mix(torch.cat([c_feat, feats[-1]], dim=1)) #4,128，64,64

        c_feat = self.nolocal(c_feat)

        K_Q, V_Q = self.keyvalue_Q(feats[-1])
        K_M, V_M = self.keyvalue_M(c_feat)
        c_feat = self.memory(K_M, V_M, K_Q, V_Q)

        differ = self.tdms(feats)

        c_feat = self.conv_fin_mix(torch.cat([c_feat,differ], dim=1))
       
        # t_feats = torch.cat([feats[j] for j in range(self.num_frame)],dim=1)
        # t_feat = self.swin(t_feats)
        # t_feat = self.conv_t(t_feat)

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
        # f_feat = self.acff(pc_feat,pc)
        
        f_feat = self.resblock2(torch.cat([pc_feat,pc], dim=1))
        
        # f_feat = self.conv_fin_mix(torch.cat([pt_feat,c_feat], dim=1))
        
        f_feats.append(f_feat)
        # f_feats = self.dyhead(f_feats)
    
        
        
        # t_feats = torch.cat([feats[j] for j in range(self.num_frame)],dim=1)
        # t_feat = self.swin(t_feats)
        # t_feat = self.conv_gl_mix(t_feat)
        
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
###TASANet
class CSWF(nn.Module):
    def __init__(self,in_channel, out_channel):
        super().__init__()
        self.conv_1 = nn.Sequential(
            BaseConv(in_channel, in_channel//2, 1, 1),
            BaseConv(in_channel//2, in_channel, 1, 1)
        )
        self.conv_2 = nn.Sequential(
            BaseConv(in_channel, in_channel//2, 1, 1),
            BaseConv(in_channel//2, out_channel, 1, 1, act="sigmoid")
        )
        self.conv = nn.Sequential(
            BaseConv(out_channel, out_channel//2, 1, 1),
            BaseConv(out_channel//2, out_channel, 1, 1)
        )
        
    def forward(self, r_feat, c_feat):
        m_feat = r_feat + c_feat
        m_feat = self.conv_2(self.conv_1(m_feat))
        m_feat = self.conv(c_feat*m_feat + r_feat*(1-m_feat))
    
        m_feat = self.conv_2(self.conv_1(m_feat))
        m_feat = self.conv(c_feat*m_feat + r_feat*(1-m_feat))
        
        return m_feat
class SimpleGate(nn.Module):
    #根据指定的维度(dim=1)将x切分为两个相等大小的部分,返回两部分的乘积
    def forward(self, x): 
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
###SEIFNet
from nets.module.CBAM import CBAM
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
    
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        a_h = a_h.expand(-1,-1,h,w)
        a_w = a_w.expand(-1, -1, h, w)

        # out = identity * a_w * a_h

        return a_w , a_h

class CoDEM2(nn.Module):
    '''
    最新的版本
    '''
    def __init__(self,channel_dim):
        super(CoDEM2, self).__init__()

        self.channel_dim=channel_dim

        #特征连接后
        self.Conv3 = nn.Conv2d(in_channels=2*self.channel_dim,out_channels=2*self.channel_dim,kernel_size=3,stride=1,padding=1)
        #特征加和后
        # self.AvgPool = nn.functional.adaptive_avg_pool2d()
        self.Conv1 = nn.Conv2d(in_channels=2*self.channel_dim,out_channels=self.channel_dim,kernel_size=1,stride=1,padding=0)
        #最后输出
        # self.Conv1_ =nn.Conv2d(in_channels=3*self.channel_dim,out_channels=self.channel_dim,kernel_size=1,stride=1,padding=0)
        self.BN1 = nn.BatchNorm2d(2*self.channel_dim)
        self.BN2 = nn.BatchNorm2d(self.channel_dim)
        self.ReLU = nn.ReLU(inplace=True)
        #我的注意力机制
        self.coAtt_1 = CoordAtt(inp=channel_dim, oup=channel_dim, reduction=16)
        #通道,kongjian注意力机制
        # self.cam =ChannelAttention(in_channels=self.channel_dim,ratio=16)
        # self.sam = SpatialAttention()

    def forward(self,x1,x2):
        B,C,H,W = x1.shape
        f_d = torch.abs(x1-x2) #B,C,H,W
        f_c = torch.cat((x1, x2), dim=1)  # B,2C,H,W
        z_c = self.ReLU(self.BN2(self.Conv1(self.ReLU(self.BN1(self.Conv3(f_c))))))

        d_aw, d_ah = self.coAtt_1(f_d)
        z_d = f_d * d_aw * d_ah


        out = z_d + z_c

        return out
class SupervisedAttentionModule(nn.Module):
    def __init__(self, mid_d):
        super(SupervisedAttentionModule, self).__init__()
        self.mid_d = mid_d

        self.cbam = CBAM(channel = self.mid_d)

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        context = self.cbam(x)

        x_out = self.conv2(context)

        return x_out
class ACFF2(nn.Module):
    '''
    最新版本的ACFF 4.21,将cat改成+，去掉卷积
    '''
    def __init__(self, channel):
        super(ACFF2, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels=channel_H,out_channels=channel_L,kernel_size=1, stride=1,padding=0)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # self.conv = nn.Conv2d(in_channels=2*channel_L, out_channels=channel_L, kernel_size=1, stride=1, padding=0)
        # self.BN = nn.BatchNorm2d(channel_L)
        # self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(in_channels=channel,ratio=16)
        # self.cbam = CBAM(channel=channel)

    def forward(self, f_low,f_high):
        # _,c,h,w = f_low.shape
        #f4上采样，通道数变成原来的1/2,长宽变为原来的2倍
        # f_high = self.relu(self.BN(self.conv1(self.up(f_high))))
        # f_high = self.relu(self.BN(self.conv1(f_high)))
        f_cat = f_high + f_low
        adaptive_w = self.ca(f_cat)
        out = f_low * adaptive_w+f_high*(1-adaptive_w) # B,C_l,h,w
        return out
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_pool_out = self.avg_pool(x)
        max_out_out = self.max_pool(x)
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool_out)))
        max_out = self.fc2(self.relu1(self.fc1(max_out_out)))
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)
###TSFNet
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
            assert torch.isnan(out).sum() == 0
            out = torch.nan_to_num(out, nan=1e-5, posinf=1e-5, neginf=1e-5)
        out = out + x
        return F.relu(out)
class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))
    def forward(self, input):
        return input * self.scale
from einops import rearrange
class MDTA(nn.Module):
    def __init__(self, dim=128, num_heads=4, bias=False):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.norm = LayerNorm2d(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x=self.norm(x)

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        return out
class selfFuseBlock(nn.Module):
    def __init__(self, channels):
        super(selfFuseBlock, self).__init__()
        self.spa = nn.Conv2d(channels, channels, 3, 1, 1)
        self.spa_att = MDTA(dim=channels)

    def forward(self, decfea, encfea=None):
        decfea = self.spa(decfea)
        decfea = self.spa_att(decfea)+decfea
        if torch.isnan(decfea).sum()>0:
            print('dec feature include NAN!!!!')
            assert torch.isnan(decfea).sum() == 0

            decfea = torch.nan_to_num(decfea, nan=1e-5, posinf=1e-5, neginf=1e-5)
        if encfea==None:
            spa=decfea
        else:
            spa=encfea + decfea
        return spa
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
class RSPB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act,norm, n_resblocks):
        super(RSPB, self).__init__()
        modules_body = [
            ResBlock_SFM(n_feat) for _ in range(n_resblocks)]

        modules_body.append(ConvBNReLU2D(n_feat, n_feat, kernel_size, padding=1, act=act, norm=norm))
        self.body = nn.Sequential(*modules_body)
        self.re_scale = Scale(1)

    def forward(self, x):
        res = self.body(x)
        return res + self.re_scale(x)
class ResBlock_SFM(nn.Module):
    def  __init__(self, num_features):
        super(ResBlock_SFM, self).__init__()
        self.layers = nn.Sequential(
            ConvBNReLU2D(num_features, out_channels=num_features, kernel_size=3, act='ReLU', padding=1),
            ConvBNReLU2D(num_features, out_channels=num_features, kernel_size=3, padding=1)
        )

    def forward(self, inputs):
        return F.relu(self.layers(inputs) + inputs)
class ConvBNReLU2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, act=None, norm=None):
        super(ConvBNReLU2D, self).__init__()

        self.layers = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.act = None
        self.norm = None
        if norm == 'BN':
            self.norm = torch.nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm = torch.nn.InstanceNorm2d(out_channels)
        elif norm == 'GN':
            self.norm = torch.nn.GroupNorm(2, out_channels)
        elif norm == 'WN':
            self.layers = torch.nn.utils.weight_norm(self.layers)
        elif norm == 'Adaptive':
            self.norm = AdaptiveNorm(n=out_channels)

        if act == 'PReLU':
            self.act = torch.nn.PReLU()
        elif act == 'SELU':
            self.act = torch.nn.SELU(True)
        elif act == 'LeakyReLU':
            self.act = torch.nn.LeakyReLU(negative_slope=0.02, inplace=True)
        elif act == 'ELU':
            self.act = torch.nn.ELU(inplace=True)
        elif act == 'ReLU':
            self.act = torch.nn.ReLU(True)
        elif act == 'Tanh':
            self.act = torch.nn.Tanh()
        elif act == 'Sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'SoftMax':
            self.act = torch.nn.Softmax2d()

    def forward(self, inputs):

        out = self.layers(inputs)

        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out
class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))

        self.bn  = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)
###F3Net
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
    
class FAD_Head(nn.Module):
    def __init__(self, size):
        super(FAD_Head, self).__init__()

        # init DCT matrix
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)

        # define base filters and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1
        low_filter = Filter(size, 0, size // 2.82)
        middle_filter = Filter(size, size // 2.82, size // 2)
        high_filter = Filter(size, size // 2, size * 2)
        all_filter = Filter(size, 0, size * 2)

        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])

    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T    # [N, 3, 299, 299]

        # 4 kernel
        y_list = []
        for i in range(4):
            x_pass = self.filters[i](x_freq)  # [N, 3, 299, 299]
            y = self._DCT_all_T @ x_pass @ self._DCT_all    # [N, 3, 299, 299]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)    # [N, 12, 299, 299]
        return out
class LFS_Head(nn.Module):
    def __init__(self, size, window_size, M):
        super(LFS_Head, self).__init__()

        self.window_size = window_size
        self._M = M

        # init DCT matrix
        self._DCT_patch = nn.Parameter(torch.tensor(DCT_mat(window_size)).float(), requires_grad=False)
        self._DCT_patch_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1), requires_grad=False)

        self.unfold = nn.Unfold(kernel_size=(window_size, window_size), stride=2, padding=4)

        # init filters
        self.filters = nn.ModuleList([Filter(window_size, window_size * 2. / M * i, window_size * 2. / M * (i+1), norm=True) for i in range(M)])
    
    def forward(self, x):
        # turn RGB into Gray
        x_gray = 0.299*x[:,0,:,:] + 0.587*x[:,1,:,:] + 0.114*x[:,2,:,:]
        x = x_gray.unsqueeze(1)

        # rescale to 0 - 255
        x = (x + 1.) * 122.5

        # calculate size
        N, C, W, H = x.size()
        S = self.window_size
        size_after = int((W - S + 8)/2) + 1
        # assert size_after == 149

        # sliding window unfold and DCT
        x_unfold = self.unfold(x)   # [N, C * S * S, L]   L:block num
        L = x_unfold.size()[2]
        x_unfold = x_unfold.transpose(1, 2).reshape(N, L, C, S, S)  # [N, L, C, S, S]
        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T

        # M kernels filtering
        y_list = []
        for i in range(self._M):
            # y = self.filters[i](x_dct)    # [N, L, C, S, S]
            # y = torch.abs(y)
            # y = torch.sum(y, dim=[2,3,4])   # [N, L]
            # y = torch.log10(y + 1e-15)
            y = torch.abs(x_dct)
            y = torch.log10(y + 1e-15)
            y = self.filters[i](y)
            y = torch.sum(y, dim=[2,3,4])
            y = y.reshape(N, size_after, size_after).unsqueeze(dim=1)   # [N, 1, 149, 149]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)  # [N, M, 149, 149]
        return out


###Octave Conv
class FirstOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(FirstOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.h2l = torch.nn.Conv2d(in_channels, int(alpha * in_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels, in_channels - int(alpha * in_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        if self.stride ==2:
            x = self.h2g_pool(x)

        X_h2l = self.h2g_pool(x) # 低频，就像GAP是DCT最低频的特殊情况？长宽缩小一半
        X_h = x
        X_h = self.h2h(X_h)
        X_l = self.h2l(X_h2l)

        return X_h, X_l
class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride
        self.l2l = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2l = torch.nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2l = self.h2g_pool(X_h)

        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)

        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)

        # X_l2h = self.upsample(X_l2h)
        X_l2h = F.interpolate(X_l2h, (int(X_h2h.size()[2]),int(X_h2h.size()[3])), mode='bilinear')
        # print('X_l2h:{}'.format(X_l2h.shape))
        # print('X_h2h:{}'.format(X_h2h.shape))
        X_h = X_l2h + X_h2h
        X_l = X_h2l + X_l2l

        return X_h, X_l

class LastOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(LastOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.l2h = torch.nn.Conv2d(int(alpha * out_channels), out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(out_channels - int(alpha * out_channels),
                                   out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, x):
        X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2h = self.h2h(X_h) # 高频组对齐通道
        X_l2h = self.l2h(X_l) # 低频组对齐通道
        # 低频组对齐长宽尺寸
        X_l2h = F.interpolate(X_l2h, (int(X_h2h.size()[2]), int(X_h2h.size()[3])), mode='bilinear')

        X_h = X_h2h + X_l2h  # 本来的设置：高频低频融合输出
        return X_h       #都输出

        # return X_h2h  #只输出高频组
        # return X_l2h    #只输出低频组

        # return X_h, X_h2h, X_l2h
class Octave(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super(Octave, self).__init__()
        # 第一层，将特征分为高频和低频
        self.fir = FirstOctaveConv(in_channels, out_channels, kernel_size)
        # 第二层，低高频输入，低高频输出
        self.mid1 = OctaveConv(in_channels, in_channels, kernel_size)
        self.mid2 = OctaveConv(in_channels, out_channels, kernel_size)
        # 第三层，将低高频汇合后输出
        self.lst = LastOctaveConv(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x0 = x
        x_h, x_l = self.fir(x)                   # (1,64,64,64) ,(1,64,32,32)
        x_hh, x_ll = x_h, x_l,
        # x_1 = x_hh +x_ll
        x_h_1, x_l_1 = self.mid1((x_h, x_l))     # (1,64,64,64) ,(1,64,32,32)
        x_h_2, x_l_2 = self.mid1((x_h_1, x_l_1)) # (1,64,64,64) ,(1,64,32,32)
        x_h_5, x_l_5 = self.mid2((x_h_2, x_l_2)) # (1,32,64,64) ,(1,32,32,32)
        x_ret = self.lst((x_h_5, x_l_5)) # (1,64,64,64)
        return x_ret

        # x_l_11 = F.interpolate(x_l_1, (int(x_h_1.size()[2]), int(x_h_1.size()[3])), mode='bilinear')
        # x_ret, x_h_6, x_l_6 = self.lst((x_h_5, x_l_5)) # (1,64,64,64)
        # return x0, x_ret,x_hh, x_ll,x_h_1, x_l_1

        # return x0, x_ret, x_hh, x_ll, x_h_6, x_l_6
        # return x0, x_ret
    # fea_name = ['_before','_after', '_beforeH', '_beforeL', '_afterH', '_afterL', '_afterH0', '_afterL0']

###Motion-memory
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
class fuse(torch.nn.Module):
    def __init__(self, indim_h, indim_l):
        super(fuse, self).__init__()
        self.conv_h = torch.nn.Conv2d(indim_h, 1, kernel_size=1)
        self.conv_l = torch.nn.Conv2d(indim_l, indim_h, kernel_size=3, padding=1, stride=1)
        self.fc = torch.nn.Linear(128, 128)

    def forward(self, l, h):
        # Eq.3 at the paper, h has been upsampled at the previous step
        S = self.conv_h(h)* self.conv_l(l) # 4 128 64 64
        temp = F.adaptive_avg_pool2d(S, (1, 1)) # 4 128 1 1
        temp = temp.squeeze(-1).squeeze(-1)
        c = self.fc(temp)
        c = c.unsqueeze(-1).unsqueeze(-1)
        return S + c * S
####ReasonLayer
class SinPositionalEncoding(nn.Module):
    """ Sinusoidal Positional Encoding"""

    def __init__(self, dim, max_seq_len):
        super(SinPositionalEncoding, self).__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('emb', emb)

    def forward(self, x):
        return self.emb[None, :x.shape[1], :].to(x)
class ReasoningLayer(nn.Module):
    """Reasoning layer IR-Reasoner"""

    def __init__(self, emb_size=128, num_heads=4, depth=1,
                 expansion_rate=1, dropout_rate=0.1, channels=[128,256,512]):
        super(ReasoningLayer, self).__init__()

        self.inp_projection = Rearrange('b e (h) (w) -> b (h w) e')

        # todo: max_seq_len will be given to reasoning layer
        self.pos_encoding = SinPositionalEncoding(emb_size, 25600)

        self.self_attn = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout_rate)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.norm1 = nn.LayerNorm(emb_size)

        self.linear1 = nn.Linear(emb_size, emb_size * expansion_rate)
        self.activation1 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.linear2 = nn.Linear(emb_size * expansion_rate, emb_size)

        self.dropout3 = nn.Dropout(p=dropout_rate)
        self.norm2 = nn.LayerNorm(emb_size)

        # self.out_projection = Rearrange('b (h w) e -> b e (h) (w)', h=spatial_dim, w=spatial_dim)

    def forward(self, src):
        # get the spatial dimension in case of different sized input  B C H W
        spatial_d1 = src.shape[2]   #输入的H
        spatial_d2 = src.shape[3]   #输入的W

        # print(src.shape)
        # alternative projections
        src = self.inp_projection(src)
        # print(src.shape)

        # use sinusodial pos encoding
        q = src + self.pos_encoding(src)
        k = src + self.pos_encoding(src)

        # Attention layer
        attn_out = self.self_attn(q, k, value=src)[0]
        add1_out = src + self.dropout1(attn_out)
        norm1_out = self.norm1(add1_out)

        # MLP layer
        mlp_out = self.linear1(norm1_out)
        mlp_out = self.activation1(mlp_out)
        mlp_out = self.dropout2(mlp_out)
        mlp_out = self.linear2(mlp_out)

        # Output norm
        add2_out = add1_out + self.dropout3(mlp_out)
        norm2_out = self.norm2(add2_out)

        # print(norm2_out.shape, spatial_d1, spatial_d2)
        reasoning_out = Rearrange('b (h w) e -> b e (h) (w)', h=spatial_d1, w=spatial_d2)(norm2_out)
        # print('****************')

        return reasoning_out
class ReasoningLayer2(nn.Module):
    """Reasoning layer  带残差"""

    def __init__(self, emb_size=128, num_heads=4, depth=1,
                 expansion_rate=1, dropout_rate=0.1, channels=[128,256,512]):
        super(ReasoningLayer2, self).__init__()

        self.inp_projection = Rearrange('b e (h) (w) -> b (h w) e')

        # todo: max_seq_len will be given to reasoning layer
        self.pos_encoding = SinPositionalEncoding(emb_size, 25600)

        self.self_attn = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout_rate)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.norm1 = nn.LayerNorm(emb_size)

        self.linear1 = nn.Linear(emb_size, emb_size * expansion_rate)
        self.activation1 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.linear2 = nn.Linear(emb_size * expansion_rate, emb_size)

        self.dropout3 = nn.Dropout(p=dropout_rate)
        self.norm2 = nn.LayerNorm(emb_size)

        self.conv_out = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )

        # self.out_projection = Rearrange('b (h w) e -> b e (h) (w)', h=spatial_dim, w=spatial_dim)

    def forward(self, src):

        srccopy = src
        # get the spatial dimension in case of different sized input  B C H W
        spatial_d1 = src.shape[2]   #输入的H
        spatial_d2 = src.shape[3]   #输入的W

        # print(src.shape)
        # alternative projections
        src = self.inp_projection(src)
        # print(src.shape)

        # use sinusodial pos encoding
        q = src + self.pos_encoding(src)
        k = src + self.pos_encoding(src)

        # Attention layer
        attn_out = self.self_attn(q, k, value=src)[0]
        add1_out = src + self.dropout1(attn_out)
        norm1_out = self.norm1(add1_out)

        # MLP layer
        mlp_out = self.linear1(norm1_out)
        mlp_out = self.activation1(mlp_out)
        mlp_out = self.dropout2(mlp_out)
        mlp_out = self.linear2(mlp_out)

        # Output norm
        add2_out = add1_out + self.dropout3(mlp_out)
        norm2_out = self.norm2(add2_out)

        # print(norm2_out.shape, spatial_d1, spatial_d2)
        reasoning_out = Rearrange('b (h w) e -> b e (h) (w)', h=spatial_d1, w=spatial_d2)(norm2_out)
        # print('****************')

        feaout = torch.concat([srccopy, reasoning_out], dim=1)

        out = self.conv_out(feaout)

        return out

if __name__ == "__main__":
    
    # from yolo_training import YOLOLoss
    # net = slowfastnet(num_classes=1, num_frame=5)
    
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
    net = ReasoningLayer2() # 4 128 64 64->4 128 64 64ccccccccc
    a = torch.randn(4, 5, 128,64,64)
    out = net(a)  
    print(out.shape)
