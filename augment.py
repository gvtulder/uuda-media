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
import elasticdeform


def augment(x, augment=None):
    # there should be one channel
    assert x.shape[0] == 1
    x_i = x[0, :, :]

    if augment:
        if 'flip' in augment:
            # flip
            t = np.random.randint(4)
            if t == 1:  # flip first dimension
                x_i = x_i[::-1, :]
            elif t == 2:  # flip second dimension
                x_i = x_i[:, ::-1]
            elif t == 3:  # flip both dimensions
                x_i = x_i[::-1, ::-1]

            # swap axes
            t = np.random.randint(2)
            if t == 1:
                x_i = np.transpose(x_i, (1, 0))

            # pytorch does not like negative strides
            x_i = x_i.copy()

        if 'rot90' in augment:
            # rotate t*90 degrees
            t = np.random.randint(4)
            for i in range(t):
                x_i = np.rot90(x_i)

            # pytorch does not like negative strides
            x_i = x_i.copy()

        if 'elastic' in augment:
            # elastic deformations
            t = np.random.randint(2)
            if t == 1:
                x_i = elasticdeform.deform_random_grid(x_i, sigma=10, points=5)

    return x_i[None, :, :]
