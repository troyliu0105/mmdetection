import torch.nn as nn
from pytorchcv.models.common import conv3x3_block, dwsconv3x3_block

from .abc_pytorchcv import BaseBackbone
from ..builder import BACKBONES
# from mmdet.models.backbones.abc_pytorchcv import BaseBackbone
# from mmdet.models.builder import BACKBONES


@BACKBONES.register_module(force=True)
class MobileNet(BaseBackbone):
    """
    MobileNet model from 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861. Also this class implements FD-MobileNet from 'FD-MobileNet: Improved MobileNet
    with A Fast Downsampling Strategy,' https://arxiv.org/abs/1802.03750.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    first_stage_stride : bool
        Whether stride is used at the first stage.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """

    def __init__(self,
                 channels=[[32], [64], [128, 128], [256, 256], [512, 512, 512, 512, 512, 512], [1024, 1024]],
                 first_stage_stride=False,
                 width_scale=1.0,
                 in_channels=3,
                 in_size=(224, 224),
                 out_indices=(3, 4, 5),
                 frozen_stages=-1,
                 num_classes=1000):
        super(MobileNet, self).__init__(out_indices, frozen_stages)
        self.name = "mobilenet_w1"
        if width_scale != 1.0:
            channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        self.in_size = in_size
        self.num_classes = num_classes
        self.features = nn.Sequential()
        init_block_channels = channels[0][0]
        self.cr_blocks = ["init_block"]
        self.features.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels[1:]):
            layer_name = "stage{}".format(i + 1)
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and ((i != 0) or first_stage_stride) else 1
                stage.add_module("unit{}".format(j + 1), dwsconv3x3_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride))
                in_channels = out_channels
            self.features.add_module(layer_name, stage)
            self.cr_blocks.append(layer_name)
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))
        self.cr_blocks.append("final_pool")
        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if 'dw_conv.conv' in name:
                nn.init.kaiming_normal_(module.weight, mode='fan_in')
            elif name == 'init_block.conv' or 'pw_conv.conv' in name:
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
            elif 'bn' in name:
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif 'output' in name:
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.cr_blocks):
            cr_block = getattr(self.features, layer_name)
            x = cr_block(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)


if __name__ == '__main__':
    def _test():
        import torch
        net = MobileNet(out_indices=(4, 5, 6))
        net.init_weights(pretrained=True)
        x = torch.rand(1, 3, 416, 416)
        out = net(x)
        for i in out:
            print(out.shape)


    _test()
