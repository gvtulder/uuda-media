#!/bin/bash
for i in $(seq 1 25) ; do
  for bal in 0.5 ; do
    for weA in 0.3 0.2 0.1 0.01 0.001 0.0001 ; do
      for weB in $weA ; do
        for delay in 0 ; do
          for lr in 0.001 0.0005 0.0001 0.00001 ; do
            for lradv in $lr ; do
              for data in BRATS BRATSInverted BRATSFlipped BRATSModalityShuffle ; do
                for model in BRATS_Conv_Linenc BRATS_Conv_Spatenc BRATS_Conv_Spatenc_EarlyJoin BRATS_Conv_Posterior ; do
                  for transform in Identity ; do
                    experiment_id=revgrad-brats2021/model-$model-data-$data-transform-$transform-cbal-$bal-advweightA-$weA-advweightB-$weB-lr-$lr-lradv-$lradv-delay-$delay
                    output_file="results/results-$experiment_id/"$( printf "%04d" $i )".npz"
                    if [ ! -f $output_file ] ; then
                       mkdir -p results/results-$experiment_id/
echo                   python -u run_experiment.py \
                       --classification-weight 0.1 \
                       --adversarial-weight-A $weA \
                       --adversarial-weight-B $weB \
                       --mb-size 128 \
                       --learning-rate-class $lr \
                       --learning-rate-adv $lradv \
                       --epochs 100 \
                       --experiment-id $i \
                       --device cuda \
                       --class-balance $bal \
                       --data $data \
                       --transform $transform \
                       --model $model \
                       --revgrad \
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
