import torch.nn as nn

from networks.building_blocks import InvertedResidual

__all__ = ['ShuffleNetV2', 'shufflenet_v2']


class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True))
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = [f'stage{i}' for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True))
        self.num_output_channels = output_channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        return x

    def forward_kd(self, x):
        raise NotImplementedError()


def shufflenet_v2(width_mult, **kwargs):
    if width_mult == 0.5:
        net = ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)
    elif width_mult == 1.0:
        net = ShuffleNetV2([4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)
    else:
        raise ValueError()
    return net
