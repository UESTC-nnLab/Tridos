import torch
import torch.nn as nn
import torch.nn.functional as F
class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out
class TDM_S(nn.Module):

    def __init__(self, nframes, apha=0.5, belta=0.5, nres_b=1):
        super(TDM_S, self).__init__()

        self.nframes = nframes
        self.apha = apha
        self.belta = belta

        base_filter = 128  # bf

        self.feat_diff = ConvBlock(128, 64, 3, 1, 1, activation='prelu', norm=None)  # 对rgb的残差信息进行特征提取：h*w*3 --> h*w*64

        self.conv1 = ConvBlock((self.nframes-1)*64, base_filter, 3, 1, 1, activation='prelu', norm=None)  # 对pooling后堆叠的diff特征增强

        # Res-Block1,h*w*bf-->h*w*64
        modules_body1 = [
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(nres_b)]
        modules_body1.append(ConvBlock(base_filter, 128, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat1 = nn.Sequential(*modules_body1)

        # Res-Block1,h*w*bf-->H*W*64，对第一次补充的目标帧特征增强
        modules_body2 = [
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(nres_b)]
        modules_body2.append(ConvBlock(base_filter, 128, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat2 = nn.Sequential(*modules_body2)

        self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)  # 池化降采样2倍

    def forward(self, feats):
        # lr_id = self.nframes // 2
        # neigbor.insert(lr_id, lr)  # 将中间目标帧插回去
        frame_list = feats
        
        rgb_diff = []
        for i in range(self.nframes-1):
            rgb_diff.append(frame_list[i] - frame_list[i+1]) #4 * [4, 128, 64, 64]

        rgb_diff = torch.stack(rgb_diff, dim=1)
        B, N, C, H, W = rgb_diff.size()  # [1, 4, 128, 64, 64]

        # 对目标帧及残差图进行特征提取
        # lr_f0 = self.feat0(lr)  # h*w*3 --> h*w*256
        lr_f0 = feats[-1]
        diff_f = self.feat_diff(rgb_diff.view(-1, C, H, W)) #16 64 64 64

        down_diff_f = self.avg_diff(diff_f).view(B, N, -1, H//2, W//2)  # [4,4,64,32,32]
        stack_diff = []
        for j in range(N):
            stack_diff.append(down_diff_f[:, j, :, :, :])
        stack_diff = torch.cat(stack_diff, dim=1) #4,256 32 32
        stack_diff = self.conv1(stack_diff)  # diff 增强1 4 128 32 32

        up_diff1 = self.res_feat1(stack_diff)  # 先过卷积256--》64再上采样 4 128 32 32

        up_diff1 = F.interpolate(up_diff1, scale_factor=2, mode='bilinear', align_corners=True)  # 使用插值上采样，通道64 # 4 128 64 64
        up_diff2 = F.interpolate(stack_diff, scale_factor=2, mode='bilinear', align_corners=True)  # 使用插值上采样，通道还是4 128 64 64

        compen_lr = self.apha * lr_f0 + self.belta * up_diff2 # 4 128 64 64

        compen_lr = self.res_feat2(compen_lr)  # 第一次补偿后增强 # 4 64 64 64

        compen_lr = self.apha * compen_lr + self.belta * up_diff1

        return compen_lr
    
class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm='batch'):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(num_filter)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(num_filter)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()


    def forward(self, x):
        residual = x
        if self.norm is not None:
            out = self.bn(self.conv1(x))
        else:
            out = self.conv1(x)

        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.bn(self.conv2(out))
        else:
            out = self.conv2(out)

        out = torch.add(out, residual)
        
        if self.activation is not None:
            out = self.act(out)
            
        return out
    
if __name__ == '__main__':
    nframes = 5
    apha = 0.5
    belta = 0.5
    nres_b = 1
    model = TDM_S(nframes, apha, belta, nres_b)
    a = torch.randn(4, 128, 64, 64)
    feats = []
    for i in range(5):
        feats.append(a)
    out = model(feats)
    print(out.shape)