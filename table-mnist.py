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

import csv

import table_util


begin_table = """
\\begin{tabularx}{\\textwidth}{X rr rr rr}
  \\toprule
        & \\multicolumn{2}{l}{Accuracy (\\%)} & \\multicolumn{2}{l}{Confidence} & \\multicolumn{1}{l}{Linear CKA} \\\\
                        & \\multicolumn{1}{l}{Source} & \\multicolumn{1}{l}{Target}
                        & \\multicolumn{1}{l}{Source} & \\multicolumn{1}{l}{Target} & \\multicolumn{1}{l}{Target} \\\\
  \\midrule
"""

end_table = """
  \\bottomrule
\\end{tabularx}
"""

columns = [
    ['classification_acc_A/validation[mean@end]', 'classification_acc_A/validation[hist@end]'],
    ['classification_acc_B/validation[mean@end]', 'classification_acc_B/validation[hist@end]'],
    ['confidence_confmat_A/validation[mean@end]', 'confidence_confmat_A/validation[hist@end]'],
    ['confidence_confmat_B/validation[mean@end]', 'confidence_confmat_B/validation[hist@end]'],
    ['linear_cka_B/validation[mean@end]',         'linear_cka_B/validation[hist@end]'],
]


HLINE = '\\addlinespace[0.6em]\\midrule'
VSPACE = '\\addlinespace[0.6em]'
rows = [
    ['MNIST, spatial encoder', None],
    ['\\hspace{1em}Unbalanced 20--80', 'results-revgrad-mnist/model-MNIST_Conv4_Spatenc-data-MNIST-transform-Identity-cbal-0.2-advweightA-0.1-advweightB-0.1-lr-0.0005-lradv-0.0005-delay-0'],
    ['\\hspace{1em}Balanced 50--50',   'results-revgrad-mnist/model-MNIST_Conv4_Spatenc-data-MNIST-transform-Identity-cbal-0.5-advweightA-0.2-advweightB-0.2-lr-0.0005-lradv-0.0005-delay-0'],
    ['\\hspace{1em}Unbalanced 80--20', 'results-revgrad-mnist/model-MNIST_Conv4_Spatenc-data-MNIST-transform-Identity-cbal-0.8-advweightA-0.01-advweightB-0.01-lr-0.0005-lradv-0.0005-delay-0'],
    VSPACE,
    ['MNIST, dense encoder', None],
    ['\\hspace{1em}Unbalanced 20--80', 'results-revgrad-mnist/model-MNIST_Conv4_Linenc-data-MNIST-transform-Identity-cbal-0.2-advweightA-0.3-advweightB-0.3-lr-0.0001-lradv-0.0001-delay-0'],
    ['\\hspace{1em}Balanced 50--50',   'results-revgrad-mnist/model-MNIST_Conv4_Linenc-data-MNIST-transform-Identity-cbal-0.5-advweightA-0.1-advweightB-0.1-lr-0.0005-lradv-0.0005-delay-0'],
    ['\\hspace{1em}Unbalanced 80--20', 'results-revgrad-mnist/model-MNIST_Conv4_Linenc-data-MNIST-transform-Identity-cbal-0.8-advweightA-0.1-advweightB-0.1-lr-0.0005-lradv-0.0005-delay-0'],
    HLINE,
    ['MNIST flipped B, spatial encoder', None],
    ['\\hspace{1em}Unbalanced 20--80', 'results-revgrad-mnist/model-MNIST_Conv4_Spatenc-data-MNISTFlipped-transform-Identity-cbal-0.2-advweightA-0.01-advweightB-0.01-lr-0.0005-lradv-0.0005-delay-0'],
    ['\\hspace{1em}Balanced 50--50',   'results-revgrad-mnist/model-MNIST_Conv4_Spatenc-data-MNISTFlipped-transform-Identity-cbal-0.5-advweightA-0.1-advweightB-0.1-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}Unbalanced 80--20', 'results-revgrad-mnist/model-MNIST_Conv4_Spatenc-data-MNISTFlipped-transform-Identity-cbal-0.8-advweightA-0.1-advweightB-0.1-lr-0.001-lradv-0.001-delay-0'],
    VSPACE,
    ['MNIST flipped B, dense encoder', None],
    ['\\hspace{1em}Unbalanced 20--80', 'results-revgrad-mnist/model-MNIST_Conv4_Linenc-data-MNISTFlipped-transform-Identity-cbal-0.2-advweightA-0.3-advweightB-0.3-lr-0.0001-lradv-0.0001-delay-0'],
    ['\\hspace{1em}Balanced 50--50',   'results-revgrad-mnist/model-MNIST_Conv4_Linenc-data-MNISTFlipped-transform-Identity-cbal-0.5-advweightA-0.1-advweightB-0.1-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}Unbalanced 80--20', 'results-revgrad-mnist/model-MNIST_Conv4_Linenc-data-MNISTFlipped-transform-Identity-cbal-0.8-advweightA-0.1-advweightB-0.1-lr-0.001-lradv-0.001-delay-0'],
    HLINE,
    ['MNIST inverted B, spatial encoder', None],
    ['\\hspace{1em}Unbalanced 20--80', 'results-revgrad-mnist/model-MNIST_Conv4_Spatenc-data-MNISTInverted-transform-Identity-cbal-0.2-advweightA-0.1-advweightB-0.1-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}Balanced 50--50',   'results-revgrad-mnist/model-MNIST_Conv4_Spatenc-data-MNISTInverted-transform-Identity-cbal-0.5-advweightA-0.3-advweightB-0.3-lr-0.0005-lradv-0.0005-delay-0'],
    ['\\hspace{1em}Unbalanced 80--20', 'results-revgrad-mnist/model-MNIST_Conv4_Spatenc-data-MNISTInverted-transform-Identity-cbal-0.8-advweightA-0.1-advweightB-0.1-lr-0.0005-lradv-0.0005-delay-0'],
    VSPACE,
    ['MNIST inverted B, dense encoder', None],
    ['\\hspace{1em}Unbalanced 20--80', 'results-revgrad-mnist/model-MNIST_Conv4_Linenc-data-MNISTInverted-transform-Identity-cbal-0.2-advweightA-0.2-advweightB-0.2-lr-0.0001-lradv-0.0001-delay-0'],
    ['\\hspace{1em}Balanced 50--50',   'results-revgrad-mnist/model-MNIST_Conv4_Linenc-data-MNISTInverted-transform-Identity-cbal-0.5-advweightA-0.2-advweightB-0.2-lr-0.0001-lradv-0.0001-delay-0'],
    ['\\hspace{1em}Unbalanced 80--20', 'results-revgrad-mnist/model-MNIST_Conv4_Linenc-data-MNISTInverted-transform-Identity-cbal-0.8-advweightA-0.2-advweightB-0.2-lr-0.0001-lradv-0.0001-delay-0'],
    HLINE,
    ['MNIST normalized, spatial encoder', None],
    ['\\hspace{1em}Unbalanced 20--80', 'results-revgrad-mnist/model-MNIST_Conv4_Spatenc-data-MNISTNormalized-transform-Identity-cbal-0.2-advweightA-0.1-advweightB-0.1-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}Balanced 50--50',   'results-revgrad-mnist/model-MNIST_Conv4_Spatenc-data-MNISTNormalized-transform-Identity-cbal-0.5-advweightA-0.1-advweightB-0.1-lr-0.0005-lradv-0.0005-delay-0'],
    ['\\hspace{1em}Unbalanced 80--20', 'results-revgrad-mnist/model-MNIST_Conv4_Spatenc-data-MNISTNormalized-transform-Identity-cbal-0.8-advweightA-0.01-advweightB-0.01-lr-0.0005-lradv-0.0005-delay-0'],
    VSPACE,
    ['MNIST normalized, dense encoder', None],
    ['\\hspace{1em}Unbalanced 20--80', 'results-revgrad-mnist/model-MNIST_Conv4_Linenc-data-MNISTNormalized-transform-Identity-cbal-0.2-advweightA-0.1-advweightB-0.1-lr-0.0005-lradv-0.0005-delay-0'],
    ['\\hspace{1em}Balanced 50--50',   'results-revgrad-mnist/model-MNIST_Conv4_Linenc-data-MNISTNormalized-transform-Identity-cbal-0.5-advweightA-0.1-advweightB-0.1-lr-0.0005-lradv-0.0005-delay-0'],
    ['\\hspace{1em}Unbalanced 80--20', 'results-revgrad-mnist/model-MNIST_Conv4_Linenc-data-MNISTNormalized-transform-Identity-cbal-0.8-advweightA-0.2-advweightB-0.2-lr-0.0001-lradv-0.0001-delay-0'],
]

experiments = {}
with open('results.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        experiments[row['experiment']] = row

table_util.render_table(begin_table, end_table, experiments, columns, rows)
