#!/bin/bash
# Script to generate BRATS data files.
#

# extract patches for each subject
cat subjects.txt | while read subject ; do
  python -u cache_brats_2015_data_per_scan.py --flatten-to-fg-bg $subject
  python -u cache_brats_2015_data_per_scan.py $subject
done

# combine subjects into subsets
for subset in trainA trainB val test ; do
  cat subjects-$subset.txt | sed 's/^/data\/brats-2015-/;s/$/-15x15x1-5000-T1-T1c-T2-Flair-labels-fgbg-normval.npz/' > data/brats-2015-15x15x1-5000-T1-T1c-T2-Flair-labels-fgbg-normval-$subset.txt
  cat subjects-$subset.txt | sed 's/^/data\/brats-2015-/;s/$/-15x15x1-5000-T1-T1c-T2-Flair-labels-normval.npz/' > data/brats-2015-15x15x1-5000-T1-T1c-T2-Flair-labels-normval-$subset.txt

  python -u cache_concatenated_data.py \
    --shuffle \
    --data-set data/brats-2015-15x15x1-5000-T1-T1c-T2-Flair-labels-fgbg-normval-$subset.txt \
    --output data/brats-2015-15x15x1-5000-T1-T1c-T2-Flair-labels-fgbg-normval-$subset.npz
  python -u cache_concatenated_data.py \
    --shuffle \
    --data-set data/brats-2015-15x15x1-5000-T1-T1c-T2-Flair-labels-normval-$subset.txt \
    --output data/brats-2015-15x15x1-5000-T1-T1c-T2-Flair-labels-normval-$subset.npz
done

