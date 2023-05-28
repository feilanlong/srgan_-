import math
import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=3,padding=1 )
        
        # 这里的in_channels和上面第一个卷积层的in_channels对应
        self.depth_conv0 = nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,padding=1,groups=channels)#groups对应out_channels
        # 这里的out_channels和上面第一个卷积层的out_channels对应
        self.point_conv0 = nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=1,padding=0)
        self.depthwise_separable_conv0 = torch.nn.Sequential(self.depth_conv0,self.point_conv0)
        # 上面3个替换了这一个卷积层

        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU(channels)
        
        # 这里的in_channels和上面第一个卷积层的in_channels对应
        self.depth_conv = nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,padding=1,groups=channels)#groups对应out_channels
        # 这里的out_channels和上面第一个卷积层的out_channels对应
        self.point_conv = nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=1,padding=0)
        self.depthwise_separable_conv = torch.nn.Sequential(self.depth_conv,self.point_conv)
        
        # 上面3个替换了这一个卷积层
        # self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        short_cut = x
        # x = self.conv1(x)
        x = self.depthwise_separable_conv0(x)


        x = self.bn1(x)
        x = self.prelu(x)

        # x = self.conv2(x)
        x = self.depthwise_separable_conv(x)
        x = self.bn2(x)

        return x + short_cut

class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class Generator(nn.Module):
    def __init__(self, scale_factor, num_residual=16):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()

        self.block_in = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU(64)
        )

        self.blocks = []
        for _ in range(num_residual):
            self.blocks.append(ResidualBlock(64))
        self.blocks = nn.Sequential(*self.blocks)
        
        self.block_out = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        self.upsample = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        self.upsample.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.upsample = nn.Sequential(*self.upsample)

    def forward(self, x):
        x = self.block_in(x)
        short_cut = x
        x = self.blocks(x)
        x = self.block_out(x)

        upsample = self.upsample(x + short_cut)
        return torch.tanh(upsample)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return self.net(x).view(batch_size)

if __name__ == "__main__":
    from torchsummary import summary

    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Generator(4).to(device)
    summary(model, input_size=(3, 16, 32))
    print("-----*********************************************-----")
    # 网络模型参数量、计算量统计
    from thop import profile, clever_format
    n_channels=3
    h=16
    w=32
    dummy_input = torch.randn(1, n_channels, h, w)
    macs, params = profile(Generator(4), inputs=(dummy_input, ), verbose=False) #, custom_ops=custom_ops) 
    macs, params = clever_format([macs, params], "%.3f")
    message = 'macs, params = ' + str(macs) + ', ' + str(params)





