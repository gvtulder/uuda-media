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

import numpy as np
import h5py

import augment

classes = {}


def register_class(cls):
    classes[cls.__name__] = cls
    return cls


###
### Synthetic, two groups
###

@register_class
class SyntheticTwo:
    # synthetic data with two classes and two clusters
    # data (0, 1)
    def __init__(self, class_balance=0.5, subset='train_A', samples=10000):
        assert subset in ('train_A', 'train_B', 'validation_A', 'validation_B')
        self.subset = subset
        self.patch_size = (10,)
        self.number_of_classes = 2
        self.number_of_groups = 2
        self.class_balance = class_balance
        self.load(subset, samples)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]
        label = self.y[idx]
        group = label
        if '_A' in self.subset:
            x = self.transform_x_for_A(x, label, group)
        else:
            x = self.transform_x_for_B(x, label, group)
        return x, label, label

    def transform_x_for_A(self, x, label, group):
        return x

    def transform_x_for_B(self, x, label, group):
        return x

    def load(self, subset, samples):
        # sample from two classes
        self.y = (np.random.rand(samples) > self.class_balance).astype(int)

        # construct samples
        self.x = np.ones((1,) + tuple(self.patch_size)) * self.y[:, None]
        # add some noise
        self.x = self.x + np.random.uniform(low=-0.5, high=0.5, size=self.x.shape)

@register_class
class SyntheticTwoReverseB(SyntheticTwo):
    # synthetic data with two classes and two clusters
    # data (0, 1) in domain A maps to (1, 0) in domain B
    def transform_x_for_B(self, x, label, group):
        # invert the intensity for domain B
        return 1 - x

@register_class
class SyntheticTwoPlusMinus(SyntheticTwo):
    # synthetic data with two classes and two clusters
    # data (-1, 1) in domain A maps to (-1, 1) in domain B
    def transform_x_for_A(self, x, label, group):
        return 2 * x - 1

    def transform_x_for_B(self, x, label, group):
        return 2 * x - 1

@register_class
class SyntheticTwoPlusMinusReverseB(SyntheticTwo):
    # synthetic data with two classes and two clusters
    # data (-1, 1) in domain A maps to (1, -1) in domain B
    def transform_x_for_A(self, x, label, group):
        return 2 * x - 1

    def transform_x_for_B(self, x, label, group):
        return 2 * (1 - x) - 1


###
### Synthetic, ten groups
###

@register_class
class SyntheticTen:
    # synthetic data with two classes and ten clusters
    # data 1000000000 = 0
    #      0100000000 = 0
    #      0010000000 = 0
    #      0001000000 = 0
    #      0000100000 = 0
    #      0000010000 = 1
    #      0000001000 = 1
    #      0000000100 = 1
    #      0000000010 = 1
    #      0000000001 = 1
    def __init__(self, class_balance=0.5, subset='train_A', samples=10000):
        assert subset in ('train_A', 'train_B', 'validation_A', 'validation_B')
        self.patch_size = (50,)
        self.number_of_classes = 2
        self.number_of_groups = 10
        self.class_balance = class_balance
        self.subset = subset
        self.load(subset, samples)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]
        group = self.y[idx]
        label = int(group >= self.class_balance * self.number_of_groups)
        if '_A' in self.subset:
            x = self.transform_x_for_A(x, label, group)
        else:
            x = self.transform_x_for_B(x, label, group)
        return x, label, group

    def transform_x_for_A(self, x, label, group):
        return x

    def transform_x_for_B(self, x, label, group):
        return x

    def load(self, subset, samples):
        # sample from 10 groups
        self.y = np.random.randint(self.number_of_groups, size=samples)

        # construct samples
        self.x = np.zeros((samples,) + tuple(self.patch_size))
        # set the labels
        for i in range(self.patch_size[0] // self.number_of_groups):
            self.x[np.arange(samples), self.y + self.number_of_groups * i] = 1

        # add some noise
        self.x = self.x + np.random.uniform(low=-0.25, high=0.25, size=self.x.shape)

@register_class
class SyntheticTenReverseB(SyntheticTen):
    # synthetic data with two classes and ten clusters
    # data for B is 1 - data for A
    def transform_x_for_B(self, x, label, group):
        # invert the intensity for domain B
        return 1 - x

@register_class
class SyntheticTenMirrorB(SyntheticTen):
    # synthetic data with two classes and ten clusters
    # groups for B are in reverse order
    def transform_x_for_B(self, x, label, group):
        # invert the intensity for domain B
        return x[::-1].copy()



###
### MNIST dataset
###

@register_class
class MNIST:
    # MNIST digits
    # 10 groups, 2 classes
    def __init__(self, class_balance=0.5, subset='train_A'):
        assert subset in ('train_A', 'train_B', 'validation_A', 'validation_B')
        self.patch_size = (1, 28, 28)
        self.number_of_classes = 2
        self.number_of_groups = 10
        self.class_balance = class_balance
        self.subset = subset
        self.load(subset)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]
        group = self.y[idx]
        label = int(group >= self.class_balance * 10)
        if '_A' in self.subset:
            x = self.transform_x_for_A(x, label, group)
        else:
            x = self.transform_x_for_B(x, label, group)
        # PyTorch does not like negative strides
        x = x.copy()
        return x, label, group

    def transform_x_for_A(self, x, label, group):
        return x

    def transform_x_for_B(self, x, label, group):
        return x

    def load(self, subset):
        # MNIST data from Keras
        # https://keras.io/api/datasets/mnist/
        with np.load('mnist.npz') as f:
            if 'train' in subset:
                self.x = f['x_train'][:, None, :, :] / 255.0
                self.y = f['y_train']
            else:
                self.x = f['x_test'][:, None, :, :] / 255.0
                self.y = f['y_test']


@register_class
class MNISTFlipped(MNIST):
    # MNIST digits
    # 10 groups, 2 classes
    # images for domain B are flipped horizontally and vertically
    # to remove spatial correspondences
    def transform_x_for_B(self, x, label, group):
        # mirror the images for domain B
        return x[:, ::-1, ::-1]


@register_class
class MNISTInverted(MNIST):
    # MNIST digits
    # 10 groups, 2 classes
    # intensities for domain B are inverted
    def transform_x_for_B(self, x, label, group):
        # invert the intensity for domain B
        return 1 - x


@register_class
class MNISTNormalized(MNIST):
    # MNIST digits
    # 10 groups, 2 classes
    # each image is normalized to mean=0, std=1
    def transform_x_for_A(self, x, label, group):
        # normalize the intensity for domain A
        return (x - np.mean(x)) / np.std(x)

    def transform_x_for_B(self, x, label, group):
        # normalize the intensity for domain B
        return (x - np.mean(x)) / np.std(x)



###
### BRATS dataset
###

@register_class
class BRATS:
    # BRATS patches
    # 4 groups, 2 classes
    def __init__(self, class_balance=0.5, subset='train_A'):
        assert subset in ('train_A', 'train_B',
                          'validation_A', 'validation_B',
                          'test_A', 'test_B')
        self.patch_size = (4, 15, 15)
        self.number_of_classes = 2
        self.number_of_groups = 4
        self.class_balance = class_balance
        self.subset = subset
        self.load(subset)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]
        group = self.y[idx]
        label = int(group >= self.class_balance * 4)
        if '_A' in self.subset:
            x = self.transform_x_for_A(x, label, group)
        else:
            x = self.transform_x_for_B(x, label, group)
        # PyTorch does not like negative strides
        x = x.copy()
        return x, label, group

    def transform_x_for_A(self, x, label, group):
        return x

    def transform_x_for_B(self, x, label, group):
        return x

    def load(self, subset):
        files = { 'train_A': 'data-brats/brats-2015-15x15x1-5000-T1-T1c-T2-Flair-labels-normval-trainA.npz',
                  'train_B': 'data-brats/brats-2015-15x15x1-5000-T1-T1c-T2-Flair-labels-normval-trainB.npz',
                  'validation_A': 'data-brats/brats-2015-15x15x1-5000-T1-T1c-T2-Flair-labels-normval-val.npz',
                  'validation_B': 'data-brats/brats-2015-15x15x1-5000-T1-T1c-T2-Flair-labels-normval-val.npz',
                  'test_A': 'data-brats/brats-2015-15x15x1-5000-T1-T1c-T2-Flair-labels-normval-test.npz',
                  'test_B': 'data-brats/brats-2015-15x15x1-5000-T1-T1c-T2-Flair-labels-normval-test.npz' }

        with np.load(files[subset]) as f:
            self.x = f['patches'][..., 0]   # go from 3D (with one slice) to 2D
            self.y = f['labels']

        # exclude normal brain (label == 1)
        self.x = self.x[self.y != 1]
        self.y = self.y[self.y != 1]

        # map classes to 0, 1, 2, 3
        # 2. necrosis
        # 3. edema
        # 4. non-enhancing tumor
        # 5. enhancing tumor
        # 1. non-tumor brain
        for l_from, l_to in zip((2, 3, 4, 5), (0, 1, 2, 3)):
            self.y[self.y == l_from] = l_to

@register_class
class BRATSInverted(BRATS):
    # BRATS patches
    # 4 groups, 2 classes
    # intensities for domain B are inverted
    def transform_x_for_B(self, x, label, group):
        return -x

@register_class
class BRATSFlipped(BRATS):
    # BRATS patches
    # 4 groups, 2 classes
    # images for domain B are flipped horizontally and vertically
    # to remove spatial correspondences
    def transform_x_for_B(self, x, label, group):
        # mirror the images for domain B
        return x[:, ::-1, ::-1]

@register_class
class BRATSModalityShuffle(BRATS):
    # BRATS patches
    # channels for domain B are reversed
    def transform_x_for_B(self, x, label, group):
        # reverse the channels for domain B
        return x[::-1, :, :]



###
### MM-WHS dataset
###

class MMWHS:
    # MMWHS patches
    # 4 groups, 2 classes
    #
    # groups:
    #   1->0 = ascending aorta (820)
    #   2->1 = left atrium blood cavity (420)
    #   3->2 = left ventricle blood cavity (500)
    #   4->3 = myocardium of the left ventricle (205)
    #
    # if class_label == 'myas-atve':
    #   labels:
    #     0 = ascending aorta + myocardium
    #     1 = left atrium + left ventricle
    #
    def __init__(self, class_balance=0.5, class_label='asmy-atve', direction='ct-mri', subset='train_A'):
        assert class_balance == 0.5
        assert subset in ('train_A', 'train_B', 'validation_A', 'validation_B')
        self.patch_size = (1, 32, 32)
        self.number_of_classes = 2
        self.number_of_groups = 4
        self.class_balance = class_balance
        self.class_label = class_label
        self.direction = direction
        self.augment = ['flip', 'rot90', 'elastic'] if 'train' in subset else None
        self.subset = subset
        self.load(subset)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]
        x = augment.augment(x, self.augment)
        if '_A' in self.subset:
            x = self.transform_x_for_A(x)
        else:
            x = self.transform_x_for_B(x)
        group = self.y[idx]
        assert self.class_label == 'asmy-atve'
        label = 0 if (group == 0 or group == 3) else 1
        return x, label, group

    def transform_x_for_A(self, x):
        return x

    def transform_x_for_B(self, x):
        return x

    def load(self, subset):
        domain_a, domain_b = self.direction.split('-')
        files = {'train_A': [f'data-mmwhs/{domain_a}_train_patches_32x32x3_subset0.h5',
                             f'data-mmwhs/{domain_a}_train_patches_32x32x3_subset1.h5'],
                 'train_B': [f'data-mmwhs/{domain_b}_train_patches_32x32x3_subset0.h5',
                             f'data-mmwhs/{domain_b}_train_patches_32x32x3_subset1.h5'],
                 'validation_A': [f'data-mmwhs/{domain_a}_train_patches_32x32x3_subset2.h5'],
                 'validation_B': [f'data-mmwhs/{domain_b}_train_patches_32x32x3_subset2.h5']}

        self.x = []
        self.y = []
        for filename in files[subset]:
            print(filename)
            with h5py.File(filename, 'r') as f:
                print(list(f), list(f['images']))
                for im in f['images'].values():
                    # take middle slice, add channel dimension
                    self.x.append(im['samples'][:, :, :, 0][:, None, :, :])
                    print(im['samples'].shape)
                    # map classes 1-4 to 0-3
                    self.y.append(im['labels'][:] - 1)
        self.x = np.concatenate(self.x, axis=0)
        self.y = np.concatenate(self.y, axis=0)

@register_class
class MMWHS_CTtoMRI(MMWHS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, direction='ct-mr', **kwargs)

@register_class
class MMWHS_MRItoCT(MMWHS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, direction='mr-ct', **kwargs)

@register_class
class MMWHS_CTtoCTinverted(MMWHS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, direction='ct-ct', **kwargs)

    # intensities for domain B are inverted
    def transform_x_for_B(self, x):
        return -x

@register_class
class MMWHS_MRItoMRIinverted(MMWHS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, direction='mr-mr', **kwargs)

    # intensities for domain B are inverted
    def transform_x_for_B(self, x):
        return -x
