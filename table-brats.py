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
\\begin{tabularx}{\\textwidth}{X rr rr r}
  \\toprule
        & \\multicolumn{2}{c}{Accuracy (\\%)} & \\multicolumn{2}{c}{Confidence} & \\multicolumn{1}{l}{Linear CKA} \\\\
                        & \\multicolumn{1}{l}{Source} & \\multicolumn{1}{l}{Target}
                        & \\multicolumn{1}{l}{Source} & \\multicolumn{1}{l}{Target} & \\multicolumn{1}{l}{Target} \\\\
  \\midrule
"""

end_table = """
  \\bottomrule
\\end{tabularx}
"""

columns = [
    ['classification_acc_A/test[mean@end]', 'classification_acc_A/test[hist@end]'],
    ['classification_acc_B/test[mean@end]', 'classification_acc_B/test[hist@end]'],
    ['confidence_confmat_A/test[mean@end]', 'confidence_confmat_A/test[hist@end]'],
    ['confidence_confmat_B/test[mean@end]', 'confidence_confmat_B/test[hist@end]'],
    ['linear_cka_B/test[mean@end]',         'linear_cka_B/test[hist@end]'],
]


HLINE = '\\addlinespace[0.6em]\\midrule'
VSPACE = '\\addlinespace[0.6em]'
rows = [
    ['BRATS, spatial encoder, early join', None],
    ['\\hspace{1em}Balanced 50--50', 'results-revgrad-brats2021bis/model-BRATS_Conv_Spatenc_EarlyJoin-data-BRATS-transform-Identity-cbal-0.5-advweightA-0.1-advweightB-0.1-lr-0.001-lradv-0.001-delay-0'],
    VSPACE,
    ['BRATS, spatial encoder', None],
    ['\\hspace{1em}Balanced 50--50', 'results-revgrad-brats2021bis/model-BRATS_Conv_Spatenc-data-BRATS-transform-Identity-cbal-0.5-advweightA-0.2-advweightB-0.2-lr-0.001-lradv-0.001-delay-0'],
    VSPACE,
    ['BRATS, dense encoder', None],
    ['\\hspace{1em}Balanced 50--50', 'results-revgrad-brats2021bis/model-BRATS_Conv_Linenc-data-BRATS-transform-Identity-cbal-0.5-advweightA-0.2-advweightB-0.2-lr-0.0001-lradv-0.0001-delay-0'],
    VSPACE,
    ['BRATS, posterior join', None],
    ['\\hspace{1em}Balanced 50--50', 'results-revgrad-brats2021bis/model-BRATS_Conv_Posterior-data-BRATS-transform-Identity-cbal-0.5-advweightA-0.3-advweightB-0.3-lr-0.0001-lradv-0.0001-delay-0'],
]

experiments = {}
with open('results.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        experiments[row['experiment']] = row

table_util.render_table(begin_table, end_table, experiments, columns, rows)
