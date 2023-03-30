#!/bin/bash
# for i in $(seq 1 50) ; do
# for i in 1 ; do
for i in $(seq 1 25) ; do
# for bal in 0.2 0.5 0.8 ; do
  for bal in 0.5 ; do
#   for weA in 0 0.5 0.2 0.1 0.01 0.001 0.0001 ; do
    for weA in 0.1 ; do
#     for weB in 0.5 0.2 0.1 0.05 0.01 0.001 0.0001 ; do
      for weB in 0.1 ; do
#       for delay in 0 20 50 ; do
        for delay in 0 ; do
          for lr in 0.001 ; do
#         for lr in 0.005 0.001 ; do
#         for lr in 0.01 ; do
#           for lradv in 0.01 0.001 0.0001 ; do
            for lradv in $lr ; do
              for data in MMWHS_CTtoMRI MMWHS_MRItoCT MMWHS_CTtoCTinverted MMWHS_MRItoMRIinverted ; do
                for model in MMWHS_Conv_Linenc MMWHS_Conv_Spatenc MMWHS_Conv_Spatenc_EarlyJoin MMWHS_Conv_Posterior ; do
                  for transform in Identity ; do
                    experiment_id=revgrad-mmwhs/model-$model-data-$data-transform-$transform-cbal-$bal-advweightA-$weA-advweightB-$weB-lr-$lr-lradv-$lradv-delay-$delay
                    output_file="results/results-$experiment_id/"$( printf "%04d" $i )".npz"
                    if [ ! -f $output_file ] ; then
                      mkdir -p results/results-$experiment_id/
echo                   python -u run_experiment.py \
                       --classification-weight 0.1 \
                       --adversarial-weight-A $weA \
                       --adversarial-weight-B $weB \
                       --mb-size 64 \
                       --learning-rate-class $lr \
                       --learning-rate-adv $lradv \
                       --epochs 150 \
                       --experiment-id $i \
                       --device cuda \
                       --class-balance $bal \
                       --data $data \
                       --transform $transform \
                       --model $model \
                       --revgrad \
                       --num-workers 4 \
                       --delay $delay \
                       --output-directory results/results-$experiment_id/ # &
                    fi
                  done
                done
              done
            done
          done
          wait
        done
      done
    done
  done
done
