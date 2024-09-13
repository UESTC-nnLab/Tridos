import torch
from torch import nn


def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace = True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = out + residual
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()
        

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        #考虑去掉一层卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out #RCU w/o CA
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out

class Res_Cor_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_Cor_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        # self.ca = ChannelAttention(out_channels)
        # self.sa = SpatialAttention()
        self.cor = CoordAtt(in_channels=out_channels,out_channels=out_channels)

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.ca(out) * out
        # out = self.sa(out) * out
        w1,w2 = self.cor(out)
        out = out * w1 *w2
        out += residual
        out = self.relu(out)
        return out
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
class DTUM(nn.Module):    # final version
    def __init__(self, in_channels, num_frames):
        super(DTUM, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0), return_indices=True)
        # self.pool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), return_indices=True, ceil_mode=False)
        self.up = nn.Upsample(scale_factor=(1,2,2), mode='nearest')
        self.relu = nn.ReLU(inplace=True)

        inch = in_channels
        pad = int((num_frames-1)/2)
        self.bn0 = nn.BatchNorm3d(inch)
        self.conv1_1 = nn.Conv3d(inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn1_1 = nn.BatchNorm3d(inch)
        self.conv2_1 = nn.Conv3d(inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn2_1 = nn.BatchNorm3d(inch)
        self.conv3_1 = nn.Conv3d(inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn3_1 = nn.BatchNorm3d(inch)
        self.conv4_1 = nn.Conv3d(inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn4_1 = nn.BatchNorm3d(inch)

        self.conv3_2 = nn.Conv3d(2*inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn3_2 = nn.BatchNorm3d(inch)
        self.conv2_2 = nn.Conv3d(2*inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn2_2 = nn.BatchNorm3d(inch)
        self.conv1_2 = nn.Conv3d(2*inch, inch, kernel_size=(num_frames,1,1), padding=(0,0,0))
        self.bn1_2 = nn.BatchNorm3d(inch)

        # self.final = nn.Sequential(
        #     nn.Conv3d(in_channels=2*inch, out_channels=32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
        #     nn.BatchNorm3d(32), nn.ReLU(),
        #     nn.Dropout3d(0.5),
        #     nn.Conv3d(in_channels=32, out_channels=num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
        # )
        self.final = nn.Sequential(nn.Conv2d(in_channels=2*inch, out_channels=2*inch,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(2*inch), nn.ReLU(),
                                   nn.Conv2d(in_channels=2*inch, out_channels= inch,kernel_size=3,stride=1,padding=1)
                                   )

    def direction(self, arr):
        b,c,t,m,n = arr.size()
        arr[:, :, 1:, :, :] = arr[:, :, 1:, :, :] - m * 2 * n * 2
        arr[:, :, 2:, :, :] = arr[:, :, 2:, :, :] - m * 2 * n * 2
        arr[:, :, 3:, :, :] = arr[:, :, 3:, :, :] - m * 2 * n * 2
        arr[:, :, 4:, :, :] = arr[:, :, 4:, :, :] - m * 2 * n * 2

        arr_r_l = arr % 2  # right 1; left 0     [0 1; 0 1]
        up_down = torch.Tensor(range(0,m)).cuda(arr.device) * n*2*2  #.transpose(0,1)
        # up_down = torch.Tensor(range(0,m)) * n*2*2  #.transpose(0,1)
        up_down = up_down.repeat_interleave(n).reshape(m,n)
        arr1 = arr.float() - up_down.reshape([1,1,1,m,n])
        arr_u_d = (arr1 >= n*2).float() * 2  # up 0; down 1  [0 0; 2 2]
        arr_out = arr_r_l.float() + arr_u_d   # [0 1; 2 3]
        arr_out = (arr_out - 1.5)       # [-1.5 -0.5; 0.5 1.5]

        return arr_out


    def forward(self, x):

        x = self.relu(self.bn0(x))

        x_1 = self.relu(self.bn1_1(self.conv1_1(x)))
        xp_1, ind = self.pool(x_1)
        x_2 = self.relu(self.bn2_1(torch.abs(self.conv2_1(xp_1 * self.direction(ind)))))
        xp_2, ind = self.pool(x_2)
        x_3 = self.relu(self.bn3_1(torch.abs(self.conv3_1(xp_2 * self.direction(ind)))))
        xp_3, ind = self.pool(x_3)
        x_4 = self.relu(self.bn4_1(torch.abs(self.conv4_1(xp_3 * self.direction(ind)))))

        o_3 = self.relu(self.bn3_2(self.conv3_2(torch.cat([self.up(x_4),x_3], dim=1))))
        o_2 = self.relu(self.bn2_2(self.conv2_2(torch.cat([self.up(o_3),x_2], dim=1)))).detach()
        o_1 = self.relu(self.bn1_2(self.conv1_2(torch.cat([self.up(o_2),x_1], dim=1))))

        x_out = torch.cat([o_1, torch.unsqueeze(x[:,:,-1,:,:],2)], dim=1) #torch.Size([4, 256, 1, 64, 64])
        x_out = self.final(x_out.squeeze(2))
        # x_out = self.final(x_out) #torch.Size([4, 1, 1, 64, 64])

        return x_out
    

if __name__ == "__main__":
    net = DTUM(128,1,1)
    a = torch.rand(4,128,64,64)
    a = a.unsqueeze(2)
    print(a.shape)
    a = net(a)
    a = a.squeeze(2)
    print(a.shape)
