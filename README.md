Unpaired, unsupervised domain adaptation assumes your domains are already similar
=================================================================================
This is the unreleased code for the experiments described in:

> **Unpaired, unsupervised domain adaptation assumes your domains are already similar**
>
> by Gijs van Tulder and Marleen de Bruijne
>
> Medical Image Analysis, 2023


Components
----------
The following scripts were used to run the experiments:

    run-all-experiments-brats.sh
    run-all-experiments-mnist.sh
    run-all-experiments-simple.sh

The tables and plots in the paper were generated with:

    render-tables.sh


Data
----
The experiments use `data.py` to load the data.

* The synthetic experiments are self-contained.
* The MNIST experiments require `mnist.npz` from the [Keras MNIST dataset](https://keras.io/api/datasets/mnist/).
* The BRATS and MM-WHS experiments require preprocessing, see the information in the `brats/` and `mmwhs/` directories.


Copyright (C) 2023 Gijs van Tulder
