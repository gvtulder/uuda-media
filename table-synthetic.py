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
        & \\multicolumn{2}{l}{} & \\multicolumn{2}{l}{Compensated} & \\multicolumn{2}{l}{} \\\\
        & \\multicolumn{2}{l}{Accuracy (\\%)} & \\multicolumn{2}{l}{accuracy (\\%)} & \\multicolumn{2}{l}{Confidence} \\\\
                        & \\multicolumn{1}{l}{Source} & \\multicolumn{1}{l}{Target}
                        & \\multicolumn{1}{l}{Source} & \\multicolumn{1}{l}{Target}
                        & \\multicolumn{1}{l}{Source} & \\multicolumn{1}{l}{Target} \\\\
  \\midrule
"""

end_table = """
  \\bottomrule
\\end{tabularx}
"""

columns = [
    ['classification_acc_A/validation[mean@end]', 'classification_acc_A/validation[hist@end]'],
    ['classification_acc_B/validation[mean@end]', 'classification_acc_B/validation[hist@end]'],
    ['^classification_acc_A/validation[mean@end]', '^classification_acc_A/validation[hist@end]'],
    ['^classification_acc_B/validation[mean@end]', '^classification_acc_B/validation[hist@end]'],
    ['confidence_confmat_A/validation[mean@end]', 'confidence_confmat_A/validation[hist@end]'],
    ['confidence_confmat_B/validation[mean@end]', 'confidence_confmat_B/validation[hist@end]'],
]


HLINE = '\\addlinespace[0.6em]\\midrule'
VSPACE = '\\addlinespace[0.6em]'
rows = [
    ['Synthetic two $0/1$', None],
    ['\\hspace{1em}Unbalanced 20--80', 'results-revgrad-simple/model-SingleDense-data-SyntheticTwo-transform-Identity-cbal-0.2-advweightA-0.5-advweightB-0.5-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}Balanced 50--50', 'results-revgrad-simple/model-SingleDense-data-SyntheticTwo-transform-Identity-cbal-0.5-advweightA-0.5-advweightB-0.5-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}Unbalanced 80--20', 'results-revgrad-simple/model-SingleDense-data-SyntheticTwo-transform-Identity-cbal-0.8-advweightA-0.5-advweightB-0.5-lr-0.001-lradv-0.001-delay-0'],
    VSPACE,
    ['Synthetic two $0/1$, inverted B', None],
    ['\\hspace{1em}Unbalanced 20--80', 'results-revgrad-simple/model-SingleDense-data-SyntheticTwoReverseB-transform-Identity-cbal-0.2-advweightA-0.5-advweightB-0.5-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}Balanced 50--50', 'results-revgrad-simple/model-SingleDense-data-SyntheticTwoReverseB-transform-Identity-cbal-0.5-advweightA-0.5-advweightB-0.5-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}Unbalanced 80--20', 'results-revgrad-simple/model-SingleDense-data-SyntheticTwoReverseB-transform-Identity-cbal-0.8-advweightA-0.5-advweightB-0.5-lr-0.001-lradv-0.001-delay-0'],
    VSPACE,
    ['Synthetic two $+1/-1$', None],
    ['\\hspace{1em}Unbalanced 20--80', 'results-revgrad-simple/model-SingleDense-data-SyntheticTwoPlusMinus-transform-Identity-cbal-0.2-advweightA-0.5-advweightB-0.5-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}Balanced 50--50', 'results-revgrad-simple/model-SingleDense-data-SyntheticTwoPlusMinus-transform-Identity-cbal-0.5-advweightA-0.5-advweightB-0.5-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}Unbalanced 80--20', 'results-revgrad-simple/model-SingleDense-data-SyntheticTwoPlusMinus-transform-Identity-cbal-0.8-advweightA-0.5-advweightB-0.5-lr-0.001-lradv-0.001-delay-0'],
    VSPACE,
    ['Synthetic two $+1/-1$, inverted B', None],
    ['\\hspace{1em}Unbalanced 20--80', 'results-revgrad-simple/model-SingleDense-data-SyntheticTwoPlusMinusReverseB-transform-Identity-cbal-0.2-advweightA-0.5-advweightB-0.5-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}Balanced 50--50', 'results-revgrad-simple/model-SingleDense-data-SyntheticTwoPlusMinusReverseB-transform-Identity-cbal-0.5-advweightA-0.5-advweightB-0.5-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}Unbalanced 80--20', 'results-revgrad-simple/model-SingleDense-data-SyntheticTwoPlusMinusReverseB-transform-Identity-cbal-0.8-advweightA-0.5-advweightB-0.5-lr-0.001-lradv-0.001-delay-0'],
    HLINE,
    ['Synthetic ten', None],
    ['\\hspace{1em}Unbalanced 20--80', 'results-revgrad-simple/model-SingleDense-data-SyntheticTen-transform-Identity-cbal-0.2-advweightA-0.5-advweightB-0.5-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}Balanced 50--50', 'results-revgrad-simple/model-SingleDense-data-SyntheticTen-transform-Identity-cbal-0.5-advweightA-0.5-advweightB-0.5-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}Unbalanced 80--20', 'results-revgrad-simple/model-SingleDense-data-SyntheticTen-transform-Identity-cbal-0.8-advweightA-0.5-advweightB-0.5-lr-0.001-lradv-0.001-delay-0'],
    VSPACE,
    ['Synthetic ten, mirrored', None],
    ['\\hspace{1em}Unbalanced 20--80', 'results-revgrad-simple/model-SingleDense-data-SyntheticTenMirrorB-transform-Identity-cbal-0.2-advweightA-0.5-advweightB-0.5-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}Balanced 50--50', 'results-revgrad-simple/model-SingleDense-data-SyntheticTenMirrorB-transform-Identity-cbal-0.5-advweightA-0.5-advweightB-0.5-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}Unbalanced 80--20', 'results-revgrad-simple/model-SingleDense-data-SyntheticTenMirrorB-transform-Identity-cbal-0.8-advweightA-0.5-advweightB-0.5-lr-0.001-lradv-0.001-delay-0'],
    VSPACE,
    ['Synthetic ten, inverted B', None],
    ['\\hspace{1em}Unbalanced 20--80', 'results-revgrad-simple/model-SingleDense-data-SyntheticTenReverseB-transform-Identity-cbal-0.2-advweightA-0.5-advweightB-0.5-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}Balanced 50--50', 'results-revgrad-simple/model-SingleDense-data-SyntheticTenReverseB-transform-Identity-cbal-0.5-advweightA-0.5-advweightB-0.5-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}Unbalanced 80--20', 'results-revgrad-simple/model-SingleDense-data-SyntheticTenReverseB-transform-Identity-cbal-0.8-advweightA-0.5-advweightB-0.5-lr-0.001-lradv-0.001-delay-0'],
]

experiments = {}
with open('results.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        experiments[row['experiment']] = row

table_util.render_table(begin_table, end_table, experiments, columns, rows)
