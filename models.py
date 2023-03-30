# Copyright (C) 2023 Gijs van Tulder
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import torch
import numpy as np

classes = {}


def register_class(cls):
    classes[cls.__name__] = cls
    return cls


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels

        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding),
            torch.nn.BatchNorm2d(self.out_channels))
        if self.in_channels != self.out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False),
                torch.nn.BatchNorm2d(self.out_channels))

    def forward(self, x):
        result = self.block(x)
        if self.in_channels != self.out_channels:
            result += self.shortcut(x)
        else:
            result += x
        return result


# helper class for two encoders
# with separate or shared weights
class _Encoders(torch.nn.Module):
    def __init__(self, encoderA, encoderB):
        super(_Encoders, self).__init__()
        self.encoderA = encoderA
        self.encoderB = encoderB

    def share_parameters(self):
        # shared initialization:
        # copy parameters from A to B
        mA = list(self.encoderA.parameters())
        mB = list(self.encoderB.parameters())
        for pA, pB in zip(mA, mB):
            pB.data[:] = pA.data[:]

    def forward(self, xA, xB):
        yA = self.encoderA(xA)
        yB = self.encoderB(xB)
        return [yA, yB]


class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def classifier_parameters(self):
        all_classifier_parameters = torch.nn.ParameterList()
        all_classifier_parameters.extend(self.encoders.parameters())
        all_classifier_parameters.extend(self.classifier.parameters())
        return all_classifier_parameters

    def discriminator_parameters(self):
        return self.discriminator.parameters()


@register_class
class SingleDense(BaseModel):
    # simple one-layer, densely connected encoders
    def __init__(self, patch_size, device):
        super(SingleDense, self).__init__()

        # network components
        # - side A
        self.encoderA = torch.nn.Sequential(
                       Flatten(),
                       torch.nn.Linear(np.product(patch_size), 10),
                   ).to(device)
        # - side B
        self.encoderB = torch.nn.Sequential(
                       Flatten(),
                       torch.nn.Linear(np.product(patch_size), 10),
                   ).to(device)

        self.encoders = _Encoders(self.encoderA, self.encoderB)

        # - shared classifier
        self.classifier = torch.nn.Sequential(
                         torch.nn.Linear(10, 1),
                     ).to(device)
        # - domain discriminator
        self.discriminator = torch.nn.Sequential(
                            torch.nn.Linear(10, 10),
                            torch.nn.ReLU(),
                            torch.nn.Linear(10, 1),
                        ).to(device)


@register_class
class MNIST_Conv4_Linenc(BaseModel):
    # convolutional network for mnist,
    # linear encoder
    def __init__(self, patch_size, device):
        super(MNIST_Conv4_Linenc, self).__init__()

        # network components
        self.encoderA, self.encoderB = [
            torch.nn.Sequential(
                torch.nn.Conv2d(1, 4, (3, 3)),  # 4 * 26x26
                torch.nn.ReLU(),
                torch.nn.Conv2d(4, 8, (3, 3)),  # 8 * 24x24
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),     # 8 * 12x12
                torch.nn.Conv2d(8, 8, (3, 3)),  # 8 * 10x10
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),     # 8 * 5x5
                Flatten(),
                torch.nn.Linear(5 * 5 * 8, 128),
                torch.nn.ReLU(),
            ).to(device)
            for i in (0, 1)]

        self.encoders = _Encoders(self.encoderA, self.encoderB)

        # - shared classifier
        self.classifier = torch.nn.Sequential(
                         torch.nn.Linear(128, 1),
                     ).to(device)
        # - domain discriminator
        self.discriminator = torch.nn.Sequential(
                            torch.nn.Linear(128, 128),
                            torch.nn.ReLU(),
                            torch.nn.Linear(128, 1),
                        ).to(device)


@register_class
class MNIST_Conv4_Linenc_BN(BaseModel):
    # convolutional network for mnist,
    # linear encoder
    def __init__(self, patch_size, device):
        super(MNIST_Conv4_Linenc_BN, self).__init__()

        # network components
        self.encoderA, self.encoderB = [
            torch.nn.Sequential(
                torch.nn.Conv2d(1, 4, (3, 3)),  # 4 * 26x26
                torch.nn.BatchNorm2d(4),
                torch.nn.ReLU(),
                torch.nn.Conv2d(4, 8, (3, 3)),  # 8 * 24x24
                torch.nn.BatchNorm2d(8),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),     # 8 * 12x12
                torch.nn.Conv2d(8, 8, (3, 3)),  # 8 * 10x10
                torch.nn.BatchNorm2d(8),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),     # 8 * 5x5
                Flatten(),
                torch.nn.Linear(5 * 5 * 8, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
            ).to(device)
            for i in (0, 1)]

        self.encoders = _Encoders(self.encoderA, self.encoderB)

        # - shared classifier
        self.classifier = torch.nn.Sequential(
                         torch.nn.Linear(128, 1),
                     ).to(device)
        # - domain discriminator
        self.discriminator = torch.nn.Sequential(
                            torch.nn.Linear(128, 128),
                            torch.nn.BatchNorm1d(128),
                            torch.nn.ReLU(),
                            torch.nn.Linear(128, 1),
                        ).to(device)


@register_class
class MNIST_Conv4_Spatenc(BaseModel):
    # convolutional network for mnist,
    # spatial inputs for linear encoder
    def __init__(self, patch_size, device):
        super(MNIST_Conv4_Spatenc, self).__init__()

        # network components
        self.encoderA, self.encoderB = [
            torch.nn.Sequential(
                torch.nn.Conv2d(1, 4, (3, 3)),  # 4 * 26x26
                torch.nn.ReLU(),
                torch.nn.Conv2d(4, 8, (3, 3)),  # 8 * 24x24
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),     # 8 * 12x12
                torch.nn.Conv2d(8, 8, (3, 3)),  # 8 * 10x10
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),     # 8 * 5x5
                Flatten(),
            ).to(device)
            for i in (0, 1)]

        self.encoders = _Encoders(self.encoderA, self.encoderB)

        # - shared classifier
        self.classifier = torch.nn.Sequential(
                         torch.nn.Linear(5 * 5 * 8, 128),
                         torch.nn.ReLU(),
                         torch.nn.Linear(128, 1),
                     ).to(device)
        # - domain discriminator
        self.discriminator = torch.nn.Sequential(
                            torch.nn.Linear(5 * 5 * 8, 128),
                            torch.nn.ReLU(),
                            torch.nn.Linear(128, 128),
                            torch.nn.ReLU(),
                            torch.nn.Linear(128, 1),
                        ).to(device)


@register_class
class MNIST_Conv4_Spatenc_BN(BaseModel):
    # convolutional network for mnist,
    # spatial inputs for linear encoder
    def __init__(self, patch_size, device):
        super(MNIST_Conv4_Spatenc, self).__init__()

        # network components
        self.encoderA, self.encoderB = [
            torch.nn.Sequential(
                torch.nn.Conv2d(1, 4, (3, 3)),  # 4 * 26x26
                torch.nn.BatchNorm2d(4),
                torch.nn.ReLU(),
                torch.nn.Conv2d(4, 8, (3, 3)),  # 8 * 24x24
                torch.nn.BatchNorm2d(8),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),     # 8 * 12x12
                torch.nn.Conv2d(8, 8, (3, 3)),  # 8 * 10x10
                torch.nn.BatchNorm2d(8),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),     # 8 * 5x5
                Flatten(),
            ).to(device)
            for i in (0, 1)]

        self.encoders = _Encoders(self.encoderA, self.encoderB)

        # - shared classifier
        self.classifier = torch.nn.Sequential(
                         torch.nn.Linear(5 * 5 * 8, 128),
                         torch.nn.BatchNorm1d(128),
                         torch.nn.ReLU(),
                         torch.nn.Linear(128, 1),
                     ).to(device)
        # - domain discriminator
        self.discriminator = torch.nn.Sequential(
                            torch.nn.Linear(5 * 5 * 8, 128),
                            torch.nn.BatchNorm1d(128),
                            torch.nn.ReLU(),
                            torch.nn.Linear(128, 128),
                            torch.nn.BatchNorm1d(128),
                            torch.nn.ReLU(),
                            torch.nn.Linear(128, 1),
                        ).to(device)


@register_class
class MNIST_Conv4_Spatenc_EarlyJoin(BaseModel):
    # convolutional network for mnist,
    # spatial inputs for linear encoder
    def __init__(self, patch_size, device):
        super(MNIST_Conv4_Spatenc_EarlyJoin, self).__init__()

        # network components
        self.encoderA, self.encoderB = [
            torch.nn.Sequential(
                torch.nn.Conv2d(1, 4, (3, 3)),  # 4 * 26x26
                torch.nn.ReLU(),
            ).to(device)
            for i in (0, 1)]

        self.encoders = _Encoders(self.encoderA, self.encoderB)

        # - shared classifier
        self.classifier = torch.nn.Sequential(
                         torch.nn.Conv2d(4, 8, (3, 3)),  # 8 * 24x24
                         torch.nn.ReLU(),
                         torch.nn.MaxPool2d((2, 2)),     # 8 * 12x12
                         torch.nn.Conv2d(8, 8, (3, 3)),  # 8 * 10x10
                         torch.nn.ReLU(),
                         torch.nn.MaxPool2d((2, 2)),     # 8 * 5x5
                         Flatten(),
                         torch.nn.Linear(5 * 5 * 8, 128),
                         torch.nn.ReLU(),
                         torch.nn.Linear(128, 1),
                     ).to(device)
        # - domain discriminator
        self.discriminator = torch.nn.Sequential(
                            torch.nn.Conv2d(4, 16, (3, 3)),  # 8 * 24x24
                            torch.nn.ReLU(),
                            torch.nn.MaxPool2d((2, 2)),     # 8 * 12x12
                            torch.nn.Conv2d(16, 32, (3, 3)),  # 8 * 10x10
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(32, 128, (1, 1)),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(128, 128, (1, 1)),
                            torch.nn.AdaptiveAvgPool2d((1, 1)),
                            Flatten(),
                            torch.nn.Linear(128, 1),
                        ).to(device)


@register_class
class BRATS_Conv_Linenc(BaseModel):
    # convolutional network for BRATS,
    # linear encoder
    def __init__(self, patch_size, device):
        super(BRATS_Conv_Linenc, self).__init__()

        # network components
        self.encoderA, self.encoderB = [
            torch.nn.Sequential(
                torch.nn.Conv2d(4, 8, (3, 3), padding=(1, 1)),     # 8 * 15x15
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 16, (4, 4)),                    # 16 * 12x12
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),                        # 16 * 6x6
                torch.nn.Conv2d(16, 32, (3, 3), padding=(1, 1)),   # 32 * 6x6
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),   # 64 * 6x6
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, (3, 3)),   # 64 * 4x4
                torch.nn.ReLU(),
                Flatten(),
                torch.nn.Linear(4 * 4 * 64, 128),
                torch.nn.ReLU(),
            ).to(device)
            for i in (0, 1)]

        self.encoders = _Encoders(self.encoderA, self.encoderB)

        # - shared classifier
        self.classifier = torch.nn.Sequential(
                         torch.nn.Linear(128, 1),
                     ).to(device)
        # - domain discriminator
        self.discriminator = torch.nn.Sequential(
                            torch.nn.Linear(128, 128),
                            torch.nn.ReLU(),
                            torch.nn.Linear(128, 1),
                        ).to(device)


@register_class
class BRATS_Conv_Spatenc(BaseModel):
    # convolutional network for BRATS,
    # linear encoder
    def __init__(self, patch_size, device):
        super(BRATS_Conv_Spatenc, self).__init__()

        # network components
        self.encoderA, self.encoderB = [
            torch.nn.Sequential(
                torch.nn.Conv2d(4, 8, (3, 3), padding=(1, 1)),     # 8 * 15x15
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 16, (4, 4)),                    # 16 * 12x12
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),                        # 16 * 6x6
                torch.nn.Conv2d(16, 32, (3, 3), padding=(1, 1)),   # 32 * 6x6
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),   # 64 * 6x6
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, (3, 3)),   # 64 * 4x4
                torch.nn.ReLU(),
                Flatten(),
            ).to(device)
            for i in (0, 1)]

        self.encoders = _Encoders(self.encoderA, self.encoderB)

        # - shared classifier
        self.classifier = torch.nn.Sequential(
                         torch.nn.Linear(4 * 4 * 64, 128),
                         torch.nn.ReLU(),
                         torch.nn.Linear(128, 1),
                     ).to(device)
        # - domain discriminator
        self.discriminator = torch.nn.Sequential(
                            torch.nn.Linear(4 * 4 * 64, 128),
                            torch.nn.ReLU(),
                            torch.nn.Linear(128, 128),
                            torch.nn.ReLU(),
                            torch.nn.Linear(128, 1),
                        ).to(device)


@register_class
class BRATS_Conv_Spatenc_EarlyJoin(BaseModel):
    # convolutional network for BRATS,
    # linear encoder
    def __init__(self, patch_size, device):
        super(BRATS_Conv_Spatenc_EarlyJoin, self).__init__()

        # network components
        self.encoderA, self.encoderB = [
            torch.nn.Sequential(
                torch.nn.Conv2d(4, 8, (3, 3), padding=(1, 1)),     # 8 * 15x15
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 16, (4, 4)),                    # 16 * 12x12
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),                        # 16 * 6x6
            ).to(device)
            for i in (0, 1)]

        self.encoders = _Encoders(self.encoderA, self.encoderB)

        # - shared classifier
        self.classifier = torch.nn.Sequential(
                         torch.nn.Conv2d(16, 32, (3, 3), padding=(1, 1)),   # 32 * 6x6
                         torch.nn.ReLU(),
                         torch.nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),   # 64 * 6x6
                         torch.nn.ReLU(),
                         torch.nn.Conv2d(64, 64, (3, 3)),   # 64 * 4x4
                         torch.nn.ReLU(),
                         Flatten(),
                         torch.nn.Linear(4 * 4 * 64, 128),
                         torch.nn.ReLU(),
                         torch.nn.Linear(128, 1),
                     ).to(device)
        # - domain discriminator
        self.discriminator = torch.nn.Sequential(
                            torch.nn.Conv2d(16, 128, (3, 3)),    # 128 * 4x4
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(128, 128, (3, 3)),   # 128 * 2x2
                            torch.nn.ReLU(),
                            Flatten(),
                            torch.nn.Linear(128 * 2 * 2, 1),
                        ).to(device)


@register_class
class BRATS_Conv_Posterior(BaseModel):
    # convolutional network for BRATS,
    # linear encoder
    def __init__(self, patch_size, device):
        super(BRATS_Conv_Posterior, self).__init__()

        # network components
        self.encoderA, self.encoderB = [
            torch.nn.Sequential(
                torch.nn.Conv2d(4, 8, (3, 3), padding=(1, 1)),     # 8 * 15x15
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 16, (4, 4)),                    # 16 * 12x12
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),                        # 16 * 6x6
                torch.nn.Conv2d(16, 32, (3, 3), padding=(1, 1)),   # 32 * 6x6
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),   # 64 * 6x6
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, (3, 3)),   # 64 * 4x4
                torch.nn.ReLU(),
                Flatten(),
                torch.nn.Linear(4 * 4 * 64, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1),
            ).to(device)
            for i in (0, 1)]

        self.encoders = _Encoders(self.encoderA, self.encoderB)

        # - shared classifier
        self.classifier = torch.nn.Sequential(
                     ).to(device)
        # - domain discriminator
        self.discriminator = torch.nn.Sequential(
                            torch.nn.Linear(1, 128),
                            torch.nn.ReLU(),
                            torch.nn.Linear(128, 1),
                        ).to(device)


@register_class
class BRATS_Conv_Posterior_BN(BaseModel):
    # convolutional network for BRATS,
    # linear encoder
    def __init__(self, patch_size, device):
        super(BRATS_Conv_Posterior_BN, self).__init__()

        # network components
        self.encoderA, self.encoderB = [
            torch.nn.Sequential(
                torch.nn.Conv2d(4, 8, (3, 3), padding=(1, 1)),     # 8 * 15x15
                torch.nn.BatchNorm2d(8),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 16, (4, 4)),                    # 16 * 12x12
                torch.nn.BatchNorm2d(16),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),                        # 16 * 6x6
                torch.nn.Conv2d(16, 32, (3, 3), padding=(1, 1)),   # 32 * 6x6
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),   # 64 * 6x6
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, (3, 3)),   # 64 * 4x4
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                Flatten(),
                torch.nn.Linear(4 * 4 * 64, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1),
            ).to(device)
            for i in (0, 1)]

        self.encoders = _Encoders(self.encoderA, self.encoderB)

        # - shared classifier
        self.classifier = torch.nn.Sequential(
                     ).to(device)
        # - domain discriminator
        self.discriminator = torch.nn.Sequential(
                            torch.nn.Linear(1, 1024),
                            torch.nn.ReLU(),
                            torch.nn.Linear(1024, 1),
                        ).to(device)


@register_class
class BRATS_Conv_Linenc_BN(BaseModel):
    # convolutional network for BRATS,
    # linear encoder
    def __init__(self, patch_size, device):
        super(BRATS_Conv_Linenc_BN, self).__init__()

        # network components
        self.encoderA, self.encoderB = [
            torch.nn.Sequential(
                torch.nn.Conv2d(4, 8, (3, 3), padding=(1, 1)),     # 8 * 15x15
                torch.nn.BatchNorm2d(8),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 16, (4, 4)),                    # 16 * 12x12
                torch.nn.BatchNorm2d(16),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),                        # 16 * 6x6
                torch.nn.Conv2d(16, 32, (3, 3), padding=(1, 1)),   # 32 * 6x6
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),   # 64 * 6x6
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, (3, 3)),   # 64 * 4x4
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                Flatten(),
                torch.nn.Linear(4 * 4 * 64, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
            ).to(device)
            for i in (0, 1)]

        self.encoders = _Encoders(self.encoderA, self.encoderB)

        # - shared classifier
        self.classifier = torch.nn.Sequential(
                         torch.nn.Linear(128, 1),
                     ).to(device)
        # - domain discriminator
        self.discriminator = torch.nn.Sequential(
                            torch.nn.Linear(128, 128),
                            torch.nn.ReLU(),
                            torch.nn.Linear(128, 1),
                        ).to(device)


@register_class
class BRATS_Conv_Spatenc_BN(BaseModel):
    # convolutional network for BRATS,
    # linear encoder
    def __init__(self, patch_size, device):
        super(BRATS_Conv_Spatenc_BN, self).__init__()

        # network components
        self.encoderA, self.encoderB = [
            torch.nn.Sequential(
                torch.nn.Conv2d(4, 8, (3, 3), padding=(1, 1)),     # 8 * 15x15
                torch.nn.BatchNorm2d(8),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 16, (4, 4)),                    # 16 * 12x12
                torch.nn.BatchNorm2d(16),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),                        # 16 * 6x6
                torch.nn.Conv2d(16, 32, (3, 3), padding=(1, 1)),   # 32 * 6x6
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),   # 64 * 6x6
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, (3, 3)),   # 64 * 4x4
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                Flatten(),
            ).to(device)
            for i in (0, 1)]

        self.encoders = _Encoders(self.encoderA, self.encoderB)

        # - shared classifier
        self.classifier = torch.nn.Sequential(
                         torch.nn.Linear(4 * 4 * 64, 128),
                         torch.nn.BatchNorm1d(128),
                         torch.nn.ReLU(),
                         torch.nn.Linear(128, 1),
                     ).to(device)
        # - domain discriminator
        self.discriminator = torch.nn.Sequential(
                            torch.nn.Linear(4 * 4 * 64, 128),
                            torch.nn.ReLU(),
                            torch.nn.Linear(128, 128),
                            torch.nn.ReLU(),
                            torch.nn.Linear(128, 1),
                        ).to(device)


@register_class
class BRATS_Conv_Spatenc_EarlyJoin_BN(BaseModel):
    # convolutional network for BRATS,
    # linear encoder
    def __init__(self, patch_size, device):
        super(BRATS_Conv_Spatenc_EarlyJoin_BN, self).__init__()

        # network components
        self.encoderA, self.encoderB = [
            torch.nn.Sequential(
                torch.nn.Conv2d(4, 8, (3, 3), padding=(1, 1)),     # 8 * 15x15
                torch.nn.BatchNorm2d(8),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 16, (4, 4)),                    # 16 * 12x12
                torch.nn.BatchNorm2d(16),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),                        # 16 * 6x6
            ).to(device)
            for i in (0, 1)]

        self.encoders = _Encoders(self.encoderA, self.encoderB)

        # - shared classifier
        self.classifier = torch.nn.Sequential(
                         torch.nn.Conv2d(16, 32, (3, 3), padding=(1, 1)),   # 32 * 6x6
                         torch.nn.BatchNorm2d(32),
                         torch.nn.ReLU(),
                         torch.nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),   # 64 * 6x6
                         torch.nn.BatchNorm2d(64),
                         torch.nn.ReLU(),
                         torch.nn.Conv2d(64, 64, (3, 3)),   # 64 * 4x4
                         torch.nn.BatchNorm2d(64),
                         torch.nn.ReLU(),
                         Flatten(),
                         torch.nn.Linear(4 * 4 * 64, 128),
                         torch.nn.BatchNorm1d(128),
                         torch.nn.ReLU(),
                         torch.nn.Linear(128, 1),
                     ).to(device)
        # - domain discriminator
        self.discriminator = torch.nn.Sequential(
                            torch.nn.Conv2d(16, 128, (3, 3)),    # 128 * 4x4
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(128, 128, (3, 3)),   # 128 * 2x2
                            torch.nn.ReLU(),
                            Flatten(),
                            torch.nn.Linear(128 * 2 * 2, 1),
                        ).to(device)


@register_class
class MMWHS_Conv_Linenc(BaseModel):
    # convolutional network for MMWHS,
    # linear encoder
    def __init__(self, patch_size, device):
        super(MMWHS_Conv_Linenc, self).__init__()

        # network components
        self.encoderA, self.encoderB = [
            torch.nn.Sequential(
                torch.nn.Conv2d(1, 4, (3, 3), padding=(1, 1)),     # 4 * 32x32
                torch.nn.ReLU(),
                torch.nn.Conv2d(4, 8, (3, 3), padding=(1, 1)),     # 8 * 32x32
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),                        # 8 * 16x16
                torch.nn.Conv2d(8, 16, (3, 3), padding=(1, 1)),    # 16 * 16x16
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, (3, 3), padding=(1, 1)),   # 32 * 16x16
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),                        # 32 * 8x8
                torch.nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),   # 64 * 8x8
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, (3, 3)),                   # 64 * 6x6
                torch.nn.ReLU(),
                Flatten(),
                torch.nn.Linear(6 * 6 * 64, 128),
                torch.nn.ReLU(),
            ).to(device)
            for i in (0, 1)]

        self.encoders = _Encoders(self.encoderA, self.encoderB)

        # - shared classifier
        self.classifier = torch.nn.Sequential(
                         torch.nn.Linear(128, 1),
                     ).to(device)
        # - domain discriminator
        self.discriminator = torch.nn.Sequential(
                            torch.nn.Linear(128, 128),
                            torch.nn.ReLU(),
                            torch.nn.Linear(128, 1),
                        ).to(device)


@register_class
class MMWHS_Conv_Spatenc(BaseModel):
    # convolutional network for MMWHS,
    # linear encoder
    def __init__(self, patch_size, device):
        super(MMWHS_Conv_Spatenc, self).__init__()

        # network components
        self.encoderA, self.encoderB = [
            torch.nn.Sequential(
                torch.nn.Conv2d(1, 4, (3, 3), padding=(1, 1)),     # 4 * 32x32
                torch.nn.ReLU(),
                torch.nn.Conv2d(4, 8, (3, 3), padding=(1, 1)),     # 8 * 32x32
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),                        # 8 * 16x16
                torch.nn.Conv2d(8, 16, (3, 3), padding=(1, 1)),    # 16 * 16x16
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, (3, 3), padding=(1, 1)),   # 32 * 16x16
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),                        # 32 * 8x8
                torch.nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),   # 64 * 8x8
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, (3, 3)),                   # 64 * 6x6
                torch.nn.ReLU(),
                Flatten(),
            ).to(device)
            for i in (0, 1)]

        self.encoders = _Encoders(self.encoderA, self.encoderB)

        # - shared classifier
        self.classifier = torch.nn.Sequential(
                         torch.nn.Linear(6 * 6 * 64, 128),
                         torch.nn.ReLU(),
                         torch.nn.Linear(128, 1),
                     ).to(device)
        # - domain discriminator
        self.discriminator = torch.nn.Sequential(
                            torch.nn.Linear(6 * 6 * 64, 128),
                            torch.nn.ReLU(),
                            torch.nn.Linear(128, 128),
                            torch.nn.ReLU(),
                            torch.nn.Linear(128, 1),
                        ).to(device)


@register_class
class MMWHS_Conv_Spatenc_EarlyJoin(BaseModel):
    # convolutional network for MMWHS,
    # linear encoder
    def __init__(self, patch_size, device):
        super(MMWHS_Conv_Spatenc_EarlyJoin, self).__init__()

        # network components
        self.encoderA, self.encoderB = [
            torch.nn.Sequential(
                torch.nn.Conv2d(1, 4, (3, 3), padding=(1, 1)),     # 4 * 32x32
                torch.nn.ReLU(),
                torch.nn.Conv2d(4, 8, (3, 3), padding=(1, 1)),     # 8 * 32x32
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),                        # 8 * 16x16
            ).to(device)
            for i in (0, 1)]

        self.encoders = _Encoders(self.encoderA, self.encoderB)

        # - shared classifier
        self.classifier = torch.nn.Sequential(
                         torch.nn.Conv2d(8, 16, (3, 3), padding=(1, 1)),    # 16 * 16x16
                         torch.nn.ReLU(),
                         torch.nn.Conv2d(16, 32, (3, 3), padding=(1, 1)),   # 32 * 16x16
                         torch.nn.ReLU(),
                         torch.nn.MaxPool2d((2, 2)),                        # 32 * 8x8
                         torch.nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),   # 64 * 8x8
                         torch.nn.ReLU(),
                         torch.nn.Conv2d(64, 64, (3, 3)),                   # 64 * 6x6
                         torch.nn.ReLU(),
                         Flatten(),
                         torch.nn.Linear(6 * 6 * 64, 128),
                         torch.nn.ReLU(),
                         torch.nn.Linear(128, 1),
                     ).to(device)
        # - domain discriminator
        self.discriminator = torch.nn.Sequential(
                            torch.nn.Conv2d(8, 16, (3, 3)),      # 16 * 16x16
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(16, 128, (3, 3)),    # 128 * 14x14
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(128, 128, (3, 3)),   # 128 * 12x12
                            torch.nn.ReLU(),
                            torch.nn.AdaptiveAvgPool2d((1, 1)),
                            Flatten(),
                            torch.nn.Linear(128, 1),
                        ).to(device)


@register_class
class MMWHS_Conv_Posterior(BaseModel):
    # convolutional network for MMWHS,
    # linear encoder
    def __init__(self, patch_size, device):
        super(MMWHS_Conv_Posterior, self).__init__()

        # network components
        self.encoderA, self.encoderB = [
            torch.nn.Sequential(
                torch.nn.Conv2d(1, 4, (3, 3), padding=(1, 1)),     # 4 * 32x32
                torch.nn.ReLU(),
                torch.nn.Conv2d(4, 8, (3, 3), padding=(1, 1)),     # 8 * 32x32
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),                        # 8 * 16x16
                torch.nn.Conv2d(8, 16, (3, 3), padding=(1, 1)),    # 16 * 16x16
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, (3, 3), padding=(1, 1)),   # 32 * 16x16
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),                        # 32 * 8x8
                torch.nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),   # 64 * 8x8
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, (3, 3)),                   # 64 * 6x6
                torch.nn.ReLU(),
                Flatten(),
                torch.nn.Linear(6 * 6 * 64, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1),
            ).to(device)
            for i in (0, 1)]

        self.encoders = _Encoders(self.encoderA, self.encoderB)

        # - shared classifier
        self.classifier = torch.nn.Sequential(
                     ).to(device)
        # - domain discriminator
        self.discriminator = torch.nn.Sequential(
                            torch.nn.Linear(1, 128),
                            torch.nn.ReLU(),
                            torch.nn.Linear(128, 1),
                        ).to(device)
