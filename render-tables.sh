#!/bin/bash
if [[ ! -f results.csv ]] ; then
  python -u tabulate-stats.py \
      --input results/results-revgrad-simple/* \
              results/results-revgrad-mnist/* \
              results/results-revgrad-brats/* \
              results/results-revgrad-mmwhs/* \
      --strip-dir-prefix '^results/' \
      --output results.csv
fi

python table-synthetic.py > tables/table-synthetic.tex
python table-simple-confmat.py > tables/table-simple-confmat.tex
python table-mnist.py > tables/table-mnist.tex
python table-brats.py > tables/table-brats.tex
python table-mmwhs.py > tables/table-mmwhs.tex
