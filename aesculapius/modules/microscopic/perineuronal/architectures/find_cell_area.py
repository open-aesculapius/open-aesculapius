import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.encoder = self.decom_resnet50()
        self.decoder = self._decoder_compositor_for_vgg16_encoder()

    @staticmethod
    def _decoder_compositor_for_vgg16_encoder():
        conv_1 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1), nn.BatchNorm2d(1024), nn.LeakyReLU()
        )
        conv_2 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU()
        )
        conv_decode_transpose_1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.BatchNorm2d(
                512), nn.LeakyReLU()
        )
        conv_decode_1 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU()
        )
        conv_decode_transpose_2 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.BatchNorm2d(
                512), nn.LeakyReLU()
        )
        conv_decode_2 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU()
        )
        conv_decode_3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.LeakyReLU()
        )
        conv_decode_transpose_3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1), nn.BatchNorm2d(
                256), nn.LeakyReLU()
        )
        conv_decode_4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.LeakyReLU()
        )
        conv_decode_5 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU()
        )
        conv_decode_transpose_4 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.BatchNorm2d(
                128), nn.LeakyReLU()
        )
        conv_decode_6 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU()
        )
        conv_decode_7 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.LeakyReLU()
        )
        conv_decode_transpose_5 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.BatchNorm2d(
                64), nn.LeakyReLU()
        )
        conv_decode_8 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.LeakyReLU()
        )
        conv_decode_9 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.LeakyReLU()
        )
        conv_decode_10 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.LeakyReLU()
        )
        conv_decode_11 = nn.Sequential(
            nn.Conv2d(16, 1, 3, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid()
        )
        return nn.Sequential(
            conv_1,
            conv_2,
            conv_decode_transpose_1,
            conv_decode_1,
            conv_decode_transpose_2,
            conv_decode_2,
            conv_decode_3,
            conv_decode_transpose_3,
            conv_decode_4,
            conv_decode_5,
            conv_decode_transpose_4,
            conv_decode_6,
            conv_decode_7,
            conv_decode_transpose_5,
            conv_decode_8,
            conv_decode_9,
            conv_decode_10,
            conv_decode_11,
        )

    @staticmethod
    def decom_resnet50():
        encoder_arch = vgg16(weights=VGG16_Weights.DEFAULT)
        filters_list = encoder_arch.features[:31]

        return filters_list

    def forward(self, data):
        e1 = self.encoder[:4](data)
        e2 = self.encoder[4:9](e1)
        e3 = self.encoder[9:16](e2)
        e4 = self.encoder[16:23](e3)
        e5 = self.encoder[23:30](e4)
        e6 = self.encoder[30](e5)
        e7 = self.decoder[:3](e6)
        x1 = F.dropout(e7, training=True)
        x1 = torch.cat([e5, x1], 1)
        x2 = self.decoder[3:5](x1)
        x2 = torch.cat([e4, x2], 1)
        x3 = self.decoder[5:8](x2)
        x3 = torch.cat([e3, x3], 1)
        x4 = self.decoder[8:11](x3)
        x4 = torch.cat([e2, x4], 1)
        x5 = self.decoder[11:](x4)
        return x5
