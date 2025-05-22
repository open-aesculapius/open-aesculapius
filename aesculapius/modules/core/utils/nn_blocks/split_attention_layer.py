import torch
import torch.nn as nn
import torch.nn.functional as F


class RSoftMax(nn.Module):
    def __init__(self, groups=1, radix=2):
        super(RSoftMax, self).__init__()

        self.groups = groups
        self.radix = radix

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, self.groups, self.radix, -1).transpose(1, 2)
        x = F.softmax(x, dim=1)
        x = x.view(B, -1, 1, 1)

        return x


class SplitAttentionLayer(nn.Module):
    """
    split attention class
    """

    def __init__(
        self,
        in_channels,
        channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=8,
        bias=True,
        radix=24,
        reduction_factor=4,
    ):
        super(SplitAttentionLayer, self).__init__()

        self.radix = radix

        self.layer_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=channels * radix,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups * radix,
                bias=bias,
            ),
            nn.BatchNorm2d(channels * radix),
            nn.ReLU(inplace=True),
        )

        inter_channels = max(32, in_channels * radix // reduction_factor)

        self.attention = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=inter_channels,
                kernel_size=1,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=inter_channels,
                out_channels=channels * radix,
                kernel_size=1,
                groups=groups,
                bias=False,
            ),
        )

        self.r_softmax = RSoftMax(groups=groups, radix=radix)

    def forward(self, x):
        x = self.layer_conv(x)

        """
        split :  [ | group 0 | group 1 | ... | group k |,  | group 0 | ... ]

        sum   :  | group 0 | group 1 | ...| group k |
        """
        B, rC = x.size()[:2]
        splits = torch.split(x, rC // self.radix, dim=1)
        gap = sum(splits)

        """
        !! becomes cardinal-major !!
        attention : |             group 0              | ... |
                    | radix 0 | radix 1| ... | radix r | ... |
        """
        att_map = self.attention(gap)

        """
        !! transposed to radix-major in rSoftMax !!
        rsoftmax : same as layer_conv
        """
        att_map = self.r_softmax(att_map)

        """
        split : same as split
        sum : same as sum
        """
        att_maps = torch.split(att_map, rC // self.radix, dim=1)
        out = sum([att_map * split for att_map,
                  split in zip(att_maps, splits)])

        """
        output : | group 0 | group 1 | ...| group k |

        concatenated tensors of all groups,
        which split attention is applied
        """

        return out.contiguous()


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
                groups=24,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x


class BottleNeck(nn.Module):
    def __init__(self, in_channels):
        super(BottleNeck, self).__init__()
        self.conv_block = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels // 64,
            kernel_size=1,
            padding=0,
            stride=1,
        )
        self.attn_layer = SplitAttentionLayer(
            in_channels=in_channels // 64,
            channels=in_channels // 64,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=8,
            bias=False,
            radix=24,
            reduction_factor=4,
        )
        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels // 64, in_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.bottleneck_block = nn.Sequential(
            self.conv_block, self.attn_layer, self.last_layer
        )

    def forward(self, x):
        z = self.bottleneck_block(x)
        return F.relu(x + z)
