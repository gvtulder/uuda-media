import glob
import os.path
import sys
import h5py

import numpy as np
import nibabel as nib


def load_patches(basename, patch_size=(32, 32, 3), samples_per_scan=1000, numpy_rng=np.random.RandomState(123)):
    print(f'Loading {basename}')
    # load files
    d_img = nib.load(f'{basename}_image.nii.gz')
    d_label = nib.load(f'{basename}_label.nii.gz')

    # load and normalize image
    x_img = d_img.get_fdata()[:]
    x_img -= x_img.mean()
    x_img /= x_img.std()

    # load and convert labels
    x_labels = d_label.dataobj[:].astype(int)
    # (1) the left ventricle blood cavity (label value 500);
    # (2) the right ventricle blood cavity (label value 600);
    # (3) the left atrium blood cavity (label value 420);
    # (4) the right atrium blood cavity (label value 550);
    # (5) the myocardium of the left ventricle (label value 205);
    # (6) the ascending aorta (label value 820);
    # (7) the pulmonary artery (label value 850).
    labels = np.zeros_like(x_labels)
    # from PnP-AdaNet/OLVA:
    labels[x_labels == 820] = 1   # (1) ascending aorta (820)
    labels[x_labels == 420] = 2   # (2) left atrium blood cavity (420)
    labels[x_labels == 500] = 3   # (3) left ventricle blood cavity (500)
    labels[x_labels == 205] = 4   # (4) myocardium of the left ventricle (205)

    # remove borders
    labels[:15, :, :] = 0
    labels[-15:, :, :] = 0
    labels[:, :15, :] = 0
    labels[:, -15:, :] = 0
    labels[:, :, :1] = 0
    labels[:, :, -1:] = 0

    # preparing lists
    labels_in_use = (1, 2, 3, 4)
    samples_per_class = samples_per_scan // len(labels_in_use)
    samples_subset_from_scan = np.zeros((len(labels_in_use) * samples_per_class, *patch_size))
    labels_subset_from_scan = np.zeros((len(labels_in_use) * samples_per_class, ), dtype=int)
    center_voxel_coordinates = np.zeros((samples_subset_from_scan.shape[0], samples_subset_from_scan.ndim - 1), dtype=int)

    sample_idx = 0
    for label in labels_in_use:
        positions = np.where(labels == label)
        subset_indices = numpy_rng.choice(len(positions[0]), min(len(positions[0]), samples_per_class), replace=False)
        sample_positions = tuple(d[subset_indices] for d in positions)
        for voxel_idx in range(sample_positions[0].shape[0]):
            voxel_coords = tuple(slice(d[voxel_idx]-patch_size[j]//2, d[voxel_idx]+patch_size[j]//2 + (patch_size[j] % 2)) for j,d in enumerate(sample_positions))
            center_voxel_coordinates[sample_idx, :] = [ d[voxel_idx] for d in sample_positions ]
            samples_subset_from_scan[sample_idx] = x_img[voxel_coords]
            labels_subset_from_scan[sample_idx] = label
            sample_idx += 1

    assert sample_idx == samples_per_scan

    return samples_subset_from_scan, labels_subset_from_scan


if __name__ == '__main__':
#   samples_subset_from_scan, labels_subset_from_scan = load_patches('images/ct_train_1001')
#   print(samples_subset_from_scan.shape)
#   print(labels_subset_from_scan.shape)

    for prefix in ('mr_train', 'ct_train'):
        filenames = sorted(list(glob.glob(f'images/{prefix}*_image.nii.gz')))
        for subset in (0, 1, 2, 3):
            with h5py.File(f'patches/{prefix}_patches_32x32x3_subset{subset}.h5', 'w') as f:
                print(f'{prefix} subset {subset}: {len(filenames)} files')
                for filename in filenames[subset::4]:
                    basename = filename.replace('_image.nii.gz', '')
                    samples_subset_from_scan, labels_subset_from_scan = \
                        load_patches(basename, samples_per_scan=5000)
                    f[f'{basename}/samples'] = samples_subset_from_scan
                    f[f'{basename}/labels'] = labels_subset_from_scan


