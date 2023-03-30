BRATS patch extraction
======================
Use these scripts to extract patches from BRATS data.

The scripts require the BRATS 2015 source files, which can be downloaded from the [CBICA Image Processing Portal](https://ipp.cbica.upenn.edu/).

Edit the files to use the correct source and target folders for your system.

Run `concat_brats_HG_2021.sh` to extract patches and generate the data files.

* `brats_data.py`: class to load scans for a BRATS subject and extract patches;
* `cache_brats_2015_data_per_scan.py`: processes a single subject and saves a patch file;
* `cache_concatenated_data.py`: combines individual subjects into subset files;
* `concat_brats_HG_2021.sh`: creates the subsets used in the paper.

-------

These files are part of the code for the paper
> **Unpaired, unsupervised domain adaptation assumes your domains are already similar**
>
> by Gijs van Tulder and Marleen de Bruijne
>
> Medical Image Analysis, 2013


Copyright (C) 2023 Gijs van Tulder
