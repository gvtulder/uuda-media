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
import numpy as np


def confmat_to_table(run_idx, confmat_A, confmat_B):
    confmat_A = (confmat_A * 100).astype(int)
    confmat_B = (confmat_B * 100).astype(int)
    line_1A = ' & '.join([('%3d' % i if i > 0 else '   ') for i in confmat_A[0]])
    line_1B = ' & '.join([('%3d' % i if i > 0 else '   ') for i in confmat_B[0]])
    line_2A = ' & '.join([('%3d' % i if i > 0 else '   ') for i in confmat_A[1]])
    line_2B = ' & '.join([('%3d' % i if i > 0 else '   ') for i in confmat_B[1]])
    s  = ' %3d & Class 1  & %s    &&   %s \\\\\n' % (run_idx, line_1A, line_1B)
    s += '     & Class 2  & %s    &&   %s \\\\\n' % (line_2A, line_2B)
    return s


output = """
\\begin{tabularx}{\\textwidth}{cl rrrrrrrrrr c rrrrrrrrrr}
  \\toprule
  Run  &  Prediction  & \\multicolumn{10}{l}{Clusters from source domain}
                     && \\multicolumn{10}{l}{Clusters from target domain} \\\\
       &              & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10
                     && 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 \\\\

"""

runs = []
runs += list(sorted(glob.glob('results/results-revgrad-simple/model-SingleDense-data-SyntheticTen-transform-Identity-cbal-0.5-advweightA-0.5-advweightB-0.5-lr-0.001-lradv-0.001-delay-0/*.npz')))[0:3]
runs += list(sorted(glob.glob('results/results-revgrad-simple/model-SingleDense-data-SyntheticTen-transform-Identity-cbal-0.2-advweightA-0.5-advweightB-0.5-lr-0.001-lradv-0.001-delay-0/*.npz')))[0:3]

for idx, run_file in enumerate(runs):

    d = np.load(run_file)
    confmat_table = confmat_to_table(idx + 1, d['confmat_A/validation'][-1],
                                     d['confmat_B/validation'][-1])
    output += '  \\midrule\n'
    output += '  %% %s\n' % run_file
    output += confmat_table + '\n'

output += "  \\bottomrule\n\\end{tabularx}"

print(output)
