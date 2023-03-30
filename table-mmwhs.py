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
    ['classification_acc_A/validation[mean@end]', 'classification_acc_A/validation[hist@end]'],
    ['classification_acc_B/validation[mean@end]', 'classification_acc_B/validation[hist@end]'],
    ['confidence_confmat_A/validation[mean@end]', 'confidence_confmat_A/validation[hist@end]'],
    ['confidence_confmat_B/validation[mean@end]', 'confidence_confmat_B/validation[hist@end]'],
    ['linear_cka_B/validation[mean@end]',         'linear_cka_B/validation[hist@end]'],
]


HLINE = '\\addlinespace[0.6em]\\midrule'
VSPACE = '\\addlinespace[0.6em]'
rows = [
    ['MM-WHS, spatial encoder, early join', None],
    ['\\hspace{1em}CT to MRI', 'results-revgrad-mmwhs/model-MMWHS_Conv_Spatenc_EarlyJoin-data-MMWHS_CTtoMRI-transform-Identity-cbal-0.5-advweightA-0.1-advweightB-0.1-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}MRI to CT', 'results-revgrad-mmwhs/model-MMWHS_Conv_Spatenc_EarlyJoin-data-MMWHS_MRItoCT-transform-Identity-cbal-0.5-advweightA-0.1-advweightB-0.1-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}CT to inverted CT', 'results-revgrad-mmwhs/model-MMWHS_Conv_Spatenc_EarlyJoin-data-MMWHS_CTtoCTinverted-transform-Identity-cbal-0.5-advweightA-0.1-advweightB-0.1-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}MRI to inverted MRI', 'results-revgrad-mmwhs/model-MMWHS_Conv_Spatenc_EarlyJoin-data-MMWHS_MRItoMRIinverted-transform-Identity-cbal-0.5-advweightA-0.1-advweightB-0.1-lr-0.001-lradv-0.001-delay-0'],
    ['MM-WHS, spatial encoder', None],
    ['\\hspace{1em}CT to MRI', 'results-revgrad-mmwhs/model-MMWHS_Conv_Spatenc-data-MMWHS_CTtoMRI-transform-Identity-cbal-0.5-advweightA-0.1-advweightB-0.1-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}MRI to CT', 'results-revgrad-mmwhs/model-MMWHS_Conv_Spatenc-data-MMWHS_MRItoCT-transform-Identity-cbal-0.5-advweightA-0.1-advweightB-0.1-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}CT to inverted CT', 'results-revgrad-mmwhs/model-MMWHS_Conv_Spatenc-data-MMWHS_CTtoCTinverted-transform-Identity-cbal-0.5-advweightA-0.1-advweightB-0.1-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}MRI to inverted MRI', 'results-revgrad-mmwhs/model-MMWHS_Conv_Spatenc-data-MMWHS_MRItoMRIinverted-transform-Identity-cbal-0.5-advweightA-0.1-advweightB-0.1-lr-0.001-lradv-0.001-delay-0'],
    ['MM-WHS, dense encoder', None],
    ['\\hspace{1em}CT to MRI', 'results-revgrad-mmwhs/model-MMWHS_Conv_Linenc-data-MMWHS_CTtoMRI-transform-Identity-cbal-0.5-advweightA-0.1-advweightB-0.1-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}MRI to CT', 'results-revgrad-mmwhs/model-MMWHS_Conv_Linenc-data-MMWHS_MRItoCT-transform-Identity-cbal-0.5-advweightA-0.1-advweightB-0.1-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}CT to inverted CT', 'results-revgrad-mmwhs/model-MMWHS_Conv_Linenc-data-MMWHS_CTtoCTinverted-transform-Identity-cbal-0.5-advweightA-0.1-advweightB-0.1-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}MRI to inverted MRI', 'results-revgrad-mmwhs/model-MMWHS_Conv_Linenc-data-MMWHS_MRItoMRIinverted-transform-Identity-cbal-0.5-advweightA-0.1-advweightB-0.1-lr-0.001-lradv-0.001-delay-0'],
    ['MM-WHS, posterior join', None],
    ['\\hspace{1em}CT to MRI', 'results-revgrad-mmwhs/model-MMWHS_Conv_Posterior-data-MMWHS_CTtoMRI-transform-Identity-cbal-0.5-advweightA-0.1-advweightB-0.1-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}MRI to CT', 'results-revgrad-mmwhs/model-MMWHS_Conv_Posterior-data-MMWHS_MRItoCT-transform-Identity-cbal-0.5-advweightA-0.1-advweightB-0.1-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}CT to inverted CT', 'results-revgrad-mmwhs/model-MMWHS_Conv_Posterior-data-MMWHS_CTtoCTinverted-transform-Identity-cbal-0.5-advweightA-0.1-advweightB-0.1-lr-0.001-lradv-0.001-delay-0'],
    ['\\hspace{1em}MRI to inverted MRI', 'results-revgrad-mmwhs/model-MMWHS_Conv_Posterior-data-MMWHS_MRItoMRIinverted-transform-Identity-cbal-0.5-advweightA-0.1-advweightB-0.1-lr-0.001-lradv-0.001-delay-0'],
]

experiments = {}
with open('results.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        experiments[row['experiment']] = row

table_util.render_table(begin_table, end_table, experiments, columns, rows)
