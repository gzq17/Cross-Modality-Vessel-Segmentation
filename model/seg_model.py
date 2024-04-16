import torch
import torch.nn as nn
from .model_utils import ResidualBlock, UpConv, DeResidualBlock

class SegmentationModel(nn.Module):

    def __init__(self, in_channel, out_channel, activate='leakrelu', norm='batch'):
        super(SegmentationModel, self).__init__()
        print(activate, norm)
        self.conv1 = ResidualBlock(in_channel, out_channels=16, stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = ResidualBlock(16, 32, stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = ResidualBlock(32, 64, stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv4 = ResidualBlock(64, 128, stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv5 = ResidualBlock(128, 256, stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)

        self.upconv1 = UpConv(256, 128, activate=activate, norm=norm)
        self.deconv1 = DeResidualBlock(256, 128, stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)

        self.upconv2 = UpConv(128, 64, activate=activate, norm=norm)
        self.deconv2 = DeResidualBlock(128, 64, stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)

        self.upconv3 = UpConv(64, 32, activate=activate, norm=norm)
        self.deconv3 = DeResidualBlock(64, 32, stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)

        self.upconv4 = UpConv(32, 16, activate=activate, norm=norm)
        self.deconv4 = DeResidualBlock(32, 16, stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)

        self.deconv5 = nn.Conv3d(16, 16, kernel_size=1, stride=1, bias=True)
        self.pred_prob = nn.Conv3d(16, out_channel, kernel_size=1, stride=1, bias=True)
        self.pred_soft = nn.Softmax(dim=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.conv5(pool4)

        deconv1_1 = self.upconv1(conv5)
        concat_1 = torch.cat((deconv1_1, conv4), dim=1)
        deconv1_2 = self.deconv1(concat_1, deconv1_1)

        deconv2_1 = self.upconv2(deconv1_2)
        concat_2 = torch.cat((deconv2_1, conv3), dim=1)
        deconv2_2 = self.deconv2(concat_2, deconv2_1)

        deconv3_1 = self.upconv3(deconv2_2)
        concat_3 = torch.cat((deconv3_1, conv2), dim=1)
        deconv3_2 = self.deconv3(concat_3, deconv3_1)

        deconv4_1 = self.upconv4(deconv3_2)
        concat_4 = torch.cat((deconv4_1, conv1), dim=1)
        deconv4_2 = self.deconv4(concat_4, deconv4_1)

        deconv5_1 = self.deconv5(deconv4_2)
        pred_prob = self.pred_prob(deconv5_1)
        pred_soft = self.pred_soft(pred_prob)

        return pred_soft

