#!/bin/bash
for i in $(seq 1 25) ; do
  for bal in 0.2 0.5 0.8 ; do
#   for weA in 0 0.5 0.2 0.1 0.01 0.001 0.0001 ; do
    for weA in 0.5 ; do
#     for weB in 0.5 0.2 0.1 0.05 0.01 0.001 0.0001 ; do
      for weB in 0.5 ; do
#       for delay in 0 20 50 ; do
        for delay in 0 ; do
#         for lr in 0.01 0.005 0.001 ; do
          for lr in 0.005 ; do
            lr=0.001
#           for lradv in 0.01 0.001 0.0001 ; do
            for lradv in 0.005 ; do
              lradv=0.001
              for data in SyntheticTwo SyntheticTwoReverseB SyntheticTwoPlusMinus SyntheticTwoPlusMinusReverseB SyntheticTen SyntheticTenReverseB SyntheticTenMirrorB ; do
                for model in SingleDense ; do
                  for transform in Identity ; do
                    experiment_id=revgrad-simple/model-$model-data-$data-transform-$transform-cbal-$bal-advweightA-$weA-advweightB-$weB-lr-$lr-lradv-$lradv-delay-$delay
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
                       --steps-per-epoch 100 \
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
        done
      done
    done
  done
done
