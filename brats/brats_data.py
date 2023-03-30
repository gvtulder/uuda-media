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

import glob
import os.path
import sys

import numpy as np
import scipy.ndimage
import dicom
import medpy.io as mio

train_scan_ids = ["HG/0001", "HG/0001"]

BASEDIR = "/archive/gvantulder/external-data/BRATS/2013/BRATS-2/Image_Data"
SCAN_TYPES = ("Flair", "T1", "T1c", "T2")

np.set_printoptions(suppress=True)


class BRATSData(object):
    def __init__(
        self,
        filter_size=(3, 3, 3),
        samples_per_scan=1000,
        numpy_rng=np.random.RandomState(123),
        scan_types=SCAN_TYPES,
        target_scan_type=None,
        load_labels=False,
        flatten_to_fg_bg=False,
        return_positions=False,
    ):
        assert target_scan_type is None or not load_labels
        self.filter_size = filter_size
        self.samples_per_scan = samples_per_scan
        self.numpy_rng = numpy_rng
        self.scan_types = scan_types
        self.target_scan_type = target_scan_type
        self.load_labels = load_labels
        self.flatten_to_fg_bg = flatten_to_fg_bg
        self.return_positions = return_positions

    def load_scans(self, scan_ids=train_scan_ids):
        collection = []

        all_scan_types = self.scan_types
        if self.target_scan_type is not None:
            all_scan_types = all_scan_types + (self.target_scan_type,)

        for i, scan_id in enumerate(scan_ids):
            print("  %2d: %s" % (i + 1, scan_id))
            scan_dir = "%s/%s" % (BASEDIR, scan_id)

            volume_pixs = []
            mask = None
            for scan_type in all_scan_types:
                print("      %s" % scan_type)
                print(scan_dir + "/*VSD.Brain*.MR_%s.*/*.mha" % scan_type)
                print(glob.glob(scan_dir + "/*VSD.Brain*.MR_%s.*/*.mha" % scan_type))
                filename = glob.glob(
                    scan_dir + "/*VSD.Brain*.MR_%s.*/*.mha" % scan_type
                )[0]
                image, header = mio.load(filename)
                volume_pixs.append(image.astype(float))
                if mask is None:
                    mask = image > 0
                else:
                    mask = mask & image > 0

            if self.load_labels:
                print("      labels")
                filename = glob.glob(scan_dir + "/*VSD.Brain*.OT.*/*.mha")[0]
                image, header = mio.load(filename)
                label_pix = image + 1
                label_pix[mask == 0] = 0

                if self.flatten_to_fg_bg:
                    # background (brain) = 1
                    # foreground (tumor) = 2
                    label_pix[label_pix > 1] = 2
                    self.max_label = 2  # np.max(label_pix)
                    labels_in_use = range(1, self.max_label + 1)
                else:
                    self.max_label = 5  # np.max(label_pix)
                    labels_in_use = range(1, self.max_label + 1)

            print("      %d voxels in mask, scan shape:" % np.sum(mask), mask.shape)

            # do not select samples near the border
            for d in range(len(self.filter_size)):
                border_start = [
                    slice(0, mask.shape[dd]) for dd in range(len(self.filter_size))
                ]
                border_start[d] = slice(0, self.filter_size[d])
                mask[tuple(border_start)] = False
                label_pix[tuple(border_start)] = 0
                border_end = [
                    slice(0, mask.shape[dd]) for dd in range(len(self.filter_size))
                ]
                border_end[d] = slice(
                    mask.shape[d] - self.filter_size[d], mask.shape[d]
                )
                mask[tuple(border_end)] = False
                label_pix[tuple(border_end)] = 0

            print("      Normalising scan to std=1, mean=0...", end="")
            for channel in xrange(len(volume_pixs)):
                mu = np.mean(volume_pixs[channel][mask])
                std = np.std(volume_pixs[channel][mask])
                volume_pixs[channel] -= mu
                volume_pixs[channel] /= std
            print("done.")

            if self.load_labels:
                samples_per_class = self.samples_per_scan / len(labels_in_use)
                print("      Selecting subset (%d per class)..." % samples_per_class)
                samples_subset_from_scan = np.zeros(
                    (len(labels_in_use) * samples_per_class, len(volume_pixs))
                    + self.filter_size
                )
                labels_subset_from_scan = np.zeros(
                    (len(labels_in_use) * samples_per_class,), dtype=int
                )
            else:
                print(
                    "      Selecting subset (%d samples per scan)..."
                    % self.samples_per_scan
                )
                samples_subset_from_scan = np.zeros(
                    (self.samples_per_scan, len(volume_pixs)) + self.filter_size
                )
                # pretend to have one 'class', True in mask, so we can use the
                # same loop to select the voxels
                samples_per_class = self.samples_per_scan
                label_pix = mask
                labels_in_use = (True,)

            center_voxel_coordinates = np.zeros(
                (samples_subset_from_scan.shape[0], samples_subset_from_scan.ndim - 2),
                dtype=int,
            )

            sample_idx = 0
            for label in labels_in_use:
                if self.load_labels:
                    print("        class %d:" % label, end="")
                positions = np.where(label_pix == label)
                if len(positions[0]) > 0:
                    subset_indices = self.numpy_rng.choice(
                        len(positions[0]),
                        min(len(positions[0]), samples_per_class),
                        replace=False,
                    )
                    sample_positions = tuple(d[subset_indices] for d in positions)
                    for voxel_idx in xrange(sample_positions[0].shape[0]):
                        voxel_coords = tuple(
                            slice(
                                d[voxel_idx] - self.filter_size[j] / 2,
                                d[voxel_idx] + self.filter_size[j] / 2 + 1,
                            )
                            for j, d in enumerate(sample_positions)
                        )
                        center_voxel_coordinates[sample_idx, :] = [
                            d[voxel_idx] for d in sample_positions
                        ]
                        for channel in xrange(len(volume_pixs)):
                            samples_subset_from_scan[sample_idx, channel] = volume_pixs[
                                channel
                            ][voxel_coords]
                            if self.load_labels:
                                labels_subset_from_scan[sample_idx] = label
                        sample_idx += 1
                print(
                    "%d of %d"
                    % (min(len(positions[0]), samples_per_class), len(positions[0]))
                )

            samples_subset_from_scan = samples_subset_from_scan[0:sample_idx]
            labels_subset_from_scan = labels_subset_from_scan[0:sample_idx]

            print("      Normalising patches to std=1, mean=0 (per source)...", end="")
            mu = np.mean(
                samples_subset_from_scan.reshape(
                    [
                        samples_subset_from_scan.shape[0],
                        samples_subset_from_scan.shape[1],
                        -1,
                    ]
                ),
                axis=2,
            )
            std = np.std(
                samples_subset_from_scan.reshape(
                    [
                        samples_subset_from_scan.shape[0],
                        samples_subset_from_scan.shape[1],
                        -1,
                    ]
                ),
                axis=2,
            )
            std[std == 0] = 1
            # keep intensity of center voxel
            intensity_subset_from_scan = samples_subset_from_scan[
                :,
                :,
                self.filter_size[0] / 2,
                self.filter_size[1] / 2,
                self.filter_size[2] / 2,
            ]
            samples_subset_from_scan -= mu.reshape([-1, len(volume_pixs), 1, 1, 1])
            samples_subset_from_scan /= std.reshape([-1, len(volume_pixs), 1, 1, 1])
            print("done.")

            if self.load_labels:
                collection.append(
                    (
                        samples_subset_from_scan,
                        labels_subset_from_scan,
                        center_voxel_coordinates,
                        intensity_subset_from_scan,
                        mu,
                        std,
                    )
                )
            elif self.target_scan_type is not None:
                collection.append(
                    (
                        samples_subset_from_scan[:, 0:-1, :, :, :],
                        samples_subset_from_scan[:, -1:, :, :, :],
                        center_voxel_coordinates,
                        intensity_subset_from_scan,
                        mu,
                        std,
                    )
                )
            else:
                collection.append(
                    (
                        samples_subset_from_scan,
                        center_voxel_coordinates,
                        intensity_subset_from_scan,
                        mu,
                        std,
                    )
                )

        if self.load_labels or self.target_scan_type is not None:
            r = []
            r.append(np.concatenate([v for v, l, c, i, m, s in collection]))
            r.append(np.concatenate([l for v, l, c, i, m, s in collection]))
            if self.return_positions:
                r.append(np.concatenate([c for v, l, c, i, m, s in collection]))
            r.append(np.concatenate([i for v, l, c, i, m, s in collection]))
            r.append(np.concatenate([m for v, l, c, i, m, s in collection]))
            r.append(np.concatenate([s for v, l, c, i, m, s in collection]))
            return tuple(r)
        else:
            r = []
            r.append(np.concatenate([v for v, c, i, m, s in collection]))
            if self.return_positions:
                r.append(np.concatenate([c for v, c, i, m, s in collection]))
            r.append(np.concatenate([i for v, c, i, m, s in collection]))
            r.append(np.concatenate([m for v, c, i, m, s in collection]))
            r.append(np.concatenate([s for v, c, i, m, s in collection]))
            return tuple(r)


if __name__ == "__main__":
    scans = BRATSData(scan_types=("T1c", "T2"), samples_per_scan=1).load_scans()
    print(scans)
    print(scans[0].shape)
