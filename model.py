import torch
import torch.nn as nn
import torchvision.models as models

class StyleTransfer(nn.Module):

    def __init__(self):
        super(StyleTransfer, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4, padding_mode="reflect")
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, padding_mode="reflect")
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, padding_mode="reflect")
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # self.convt1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.convt1 = UpScale(128, 64, kernel_size=3, stride=1, padding=1, upscale_factor=2)
        self.in4 = nn.InstanceNorm2d(64)
        # self.convt2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.convt2 = UpScale(64, 32, kernel_size=3, stride=1, padding=1, upscale_factor=2)
        self.in5 = nn.InstanceNorm2d(32)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4, padding_mode="reflect")

        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.in1(self.conv1(x)))
        x = self.relu(self.in2(self.conv2(x)))
        x = self.relu(self.in3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.relu(self.in4(self.convt1(x)))
        x = self.relu(self.in5(self.convt2(x)))
        x = self.conv4(x)
        return x

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.in1(self.conv1(x)))
        x = self.in2(self.conv2(x))
        x = x + residual
        return x

class UpScale(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, upscale_factor):
        super(UpScale, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode="reflect")

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='nearest')
        return self.conv(x)

class VGG(nn.Module):

    def __init__(self):
        super(VGG, self).__init__()
        self.layers = [0, 5, 10, 17, 24]
        self.vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:self.layers[-1]+1]

    def forward(self, x):
        features = []
        for idx, layer in enumerate(self.vgg):
            x = layer(x)

            if idx in self.layers:
                features.append(x)
        return features
