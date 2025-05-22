import torch
from torch import nn

from aesculapius.modules.core.utils.nn_blocks.split_attention_layer import (
    BottleNeck)
from aesculapius.modules.core.utils.nn_blocks import vgg_block as blocks


class TransformerVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = blocks.decom_resnet50()
        self.blocks_layer_1 = BottleNeck(2048 * 24)
        self.blocks_layer_2 = BottleNeck(2048 * 24)
        self.bn_f = nn.BatchNorm2d(2048 * 24)
        self.lm_head = nn.Conv2d(
            2048 * 24, 1, kernel_size=1, padding=0, stride=1)
        self.weight_init(0, 0.02)
        self.bce_with_w = nn.BCEWithLogitsLoss(reduction="none")

    def weight_init(self, mean, std):
        for param in self.encoder:
            if isinstance(param, nn.ConvTranspose2d) or isinstance(
                    param, nn.Conv2d):
                param.weight.requires_grad = False

    def forward(self, x, targets=None, weights=None):
        B, Num_im, C, dim_1, dim_2 = x.shape
        x = x.contiguous().view((-1, C, dim_1, dim_2))
        x = self.encoder(x)

        _, C, dim_1, dim_2 = x.shape
        x = x.view((-1, Num_im, C, dim_1, dim_2))
        B, _, _, dim_1, dim_2 = x.shape
        x = x.contiguous().view(B, -1, dim_1, dim_2)

        x = self.blocks_layer_1(x)
        x = self.blocks_layer_2(x)
        x = self.bn_f(x)
        x = self.lm_head(x)

        if targets is None or weights is None:
            loss = None
        else:
            loss = torch.sum(
                torch.diagonal(
                    torch.transpose(
                        weights, -1, -2) @ self.bce_with_w(x, targets),
                    dim1=2,
                    dim2=3,
                )
            )
            assert loss is not None
            loss = loss / B
        return x, loss


def collate_fn(batch):
    return tuple(zip(*batch))
