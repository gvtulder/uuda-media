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

import argparse
import os
import sys
import numpy as np
import brats_data

brats_data.BASEDIR = (
    "/scratch/gvantulder/external-data/BRATS/2015/BRATS2015_Training/HGG"
)

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "scans", metavar="SCAN", nargs="+", help="BRATS subject IDs to process"
)
parser.add_argument(
    "--flatten-to-fg-bg",
    action="store_true",
    default=False,
    help="use binary labels (tumor vs background) instead of tumor labels",
)
args = parser.parse_args()

scan_ids = args.scans

filter_size = (15, 15, 1)
samples_per_scan = 5000
restriction = None
src = ("T1", "T1c", "T2", "Flair")
load_labels = True
flatten_to_fg_bg = args.flatten_to_fg_bg

param_string = (
    "%dx%dx%d-%d-%s" % (filter_size + (samples_per_scan,) + ("-".join(src),))
    + ("-labels" if load_labels else "")
    + ("-fgbg" if flatten_to_fg_bg else "")
    + "-normval"
)

if restriction is not None:
    param_string = "%s-%s" % (param_string, restriction)
    d = brats_data.BRATSData(
        filter_size=filter_size,
        samples_per_scan=samples_per_scan,
        numpy_rng=np.random.RandomState(123),
        scan_types=src,
        load_labels=load_labels,
        return_positions=True,
        flatten_to_fg_bg=flatten_to_fg_bg,
    )
else:
    d = brats_data.BRATSData(
        filter_size=filter_size,
        samples_per_scan=samples_per_scan,
        numpy_rng=np.random.RandomState(123),
        scan_types=src,
        load_labels=load_labels,
        return_positions=True,
        flatten_to_fg_bg=flatten_to_fg_bg,
    )

for scan_id in scan_ids:
    scan_id_for_filename = scan_id.replace("/", "")
    filename = "data/brats-2015-%s-%s.npz" % (scan_id_for_filename, param_string)
    if not os.path.exists(filename):
        print("scan_id: %s" % scan_id)
        if load_labels:
            patches, labels, voxel_coords, intensity, mu, std = d.load_scans([scan_id])
            print("patches.shape", patches.shape)
            print("labels.shape", labels.shape)
            print("voxel_coords.shape", voxel_coords.shape)
            print("intensity.shape", intensity.shape)
            print("mu.shape", mu.shape)
            print("std.shape", std.shape)
            np.savez_compressed(
                filename,
                patches=patches,
                voxel_coords=voxel_coords,
                intensity=intensity,
                labels=labels,
                mu=mu,
                std=std,
            )
        else:
            patches, voxel_coords, intensity, mu, std = d.load_scans([scan_id])
            print("patches.shape", patches.shape)
            print("voxel_coords.shape", voxel_coords.shape)
            print("intensity.shape", intensity.shape)
            print("mu.shape", mu.shape)
            print("std.shape", std.shape)
            np.savez_compressed(
                filename,
                patches=patches,
                voxel_coords=voxel_coords,
                intensity=intensity,
                mu=mu,
                std=std,
            )
