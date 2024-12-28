import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3, *args, **kwargs):
        """Classic ResNet building block using 2 convolutions with batchnorm and non-linearity.

        Args:
            c_in (int): number of in channels
            c_out (int): number of out channel
            kernel_size (int, optional): size of conv kernel. Defaults to 3.
        """
        super().__init__(*args, **kwargs)

        # If num channels not match, make this a downsampling with stride 2 (typically groups first layer)
        self.downsample = None
        if c_out != c_in:
            self.downsample = nn.Conv2d(
                c_in, c_out, kernel_size=1, stride=2, bias=False
            )

        self.activation = nn.ReLU()
        self.block = nn.Sequential(
            nn.Conv2d(
                c_in,
                c_out,
                kernel_size,
                padding=1,
                bias=False,
                stride=2 if self.downsample else 1,
            ),
            nn.BatchNorm2d(c_out),
            self.activation,
            nn.Conv2d(c_out, c_out, kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        )

    def forward(self, x: torch.Tensor):
        x2 = self.block(x)
        if self.downsample:
            x = self.downsample(x)
        out = x + x2
        out = self.activation(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        num_blocks: list[int] = [3, 3, 3],
        c_hidden: list[int] = [16, 32, 64],
        num_classes: int = 10,
        *args,
        **kwargs
    ):
        """Full Resnet network architecture.

        Args:
            num_blocks (list, optional): number of blocks per group. Defaults to [3, 3, 3].
            c_hidden (list, optional): hidden channel dim per group. Defaults to [16, 32, 64].
            num_classes (int, optional): number of output classes. Defaults to 10.
        """
        super().__init__(*args, **kwargs)

        self.input = nn.Sequential(
            nn.Conv2d(3, c_hidden[0], 3, padding=1, stride=1),
            nn.BatchNorm2d(c_hidden[0]),
            nn.ReLU(),
        )

        self.groups = nn.Sequential()
        for i, (num_block, c_in) in enumerate(zip(num_blocks, c_hidden)):
            group = nn.Sequential()
            for j in range(num_block):

                # Add downsample layer at group beginning
                if i > 0 and j == 0:
                    group.append(ResNetBlock(c_in=c_in // 2, c_out=c_in))
                else:
                    group.append(ResNetBlock(c_in=c_in, c_out=c_in))
            self.groups += group

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c_hidden[-1], num_classes),
        )

        self._init_parameters()

    def _init_parameters(self):
        """Init model parameters with Kaiming-He or constant."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor):
        x = self.input(x)
        x = self.groups(x)
        out = self.output(x)
        return out
