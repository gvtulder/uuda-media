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

import sys
import numpy as np
import helpers

import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("--output", metavar="FILE", required=True)
parser.add_argument(
    "--channel-mu-std-to-zero",
    metavar="CHANNEL",
    type=int,
    help="set mu, std for CHANNEL to zero",
)
parser.add_argument("--data-set", metavar="FILE", help="npz", required=True)
parser.add_argument("--shuffle", action="store_true")
parser.add_argument("--shuffle-seed", metavar="SEED", type=int)
args = parser.parse_args()

patches, labels, intensity, mu, std = helpers.load_data(
    args.data_set, patches_dtype="float32", return_mu_and_std=True
)

if args.channel_mu_std_to_zero is not None:
    intensity[:, args.channel_mu_std_to_zero] = 0
    mu[:, args.channel_mu_std_to_zero] = 0
    std[:, args.channel_mu_std_to_zero] = 1

if args.shuffle:
    rstate = np.random.RandomState(seed=args.shuffle_seed)
    shuffle_order = rstate.permutation(patches.shape[0])
    patches = patches[shuffle_order]
    if labels is not None:
        labels = labels[shuffle_order]
    if intensity is not None:
        intensity = intensity[shuffle_order]
    if mu is not None:
        mu = mu[shuffle_order]
    if std is not None:
        std = std[shuffle_order]

data = {}

print("patches.shape", patches.shape)
data["patches"] = patches
if labels is not None:
    print("labels.shape", labels.shape)
    data["labels"] = labels
if intensity is not None:
    print("intensity.shape", intensity.shape)
    data["intensity"] = intensity
if mu is not None:
    print("mu.shape", mu.shape)
    data["mu"] = mu
if std is not None:
    print("std.shape", std.shape)
    data["std"] = std

if args.shuffle:
    data["preshuffled"] = args.shuffle
if args.shuffle_seed:
    data["shuffle_seed"] = args.shuffle_seed

np.savez_compressed(args.output, **data)
