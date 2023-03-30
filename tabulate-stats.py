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

import argparse
import glob
import json
import re
import csv
import collections
import numpy as np

np.set_printoptions(suppress=True, linewidth=160)

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='input directories', required=True, nargs='+')
parser.add_argument('--html-table', action='store_true')
parser.add_argument('--text-output', action='store_true')
parser.add_argument('--strip-dir-prefix')
parser.add_argument('--output', help='output file')
parser.add_argument('--trim-epochs', metavar='EPOCHS', type=int, help='ignore epochs >= EPOCHS ')
args = parser.parse_args()

lines = []
keys = set()

for input_dir in args.input:
    # load results for individual experiments
    line_results = collections.defaultdict(lambda: [])
    for input_file in glob.glob('%s/*.npz' % input_dir):
        with open(input_file.replace('.npz', '-params.json'), 'r') as f:
            experiment_params = json.load(f)
        with np.load(input_file) as d:
            for key in sorted(d.keys()):
                if d[key].dtype not in ('float32', 'float64'):
                    print('Invalid dtype %s for key %s in %s' % (d[key].dtype, key, input_file))
                    continue
                if 'confmat' in key:
                    # old confidence metric
                    # (sensitive to class balance)
                    diff_class = np.sum(d[key][:, 0, :], axis=-1) - np.sum(d[key][:, 1, :], axis=-1)
                    confidence = np.sum(np.max(d[key], axis=1), axis=-1) - np.abs(diff_class)
                    line_results['abs_diff_class_%s' % key].append(np.abs(diff_class))
                    line_results['diff_class_%s' % key].append(diff_class)
                    line_results['confidence_old_%s' % key].append(confidence)

                    # new confidence metric
                    # (not sensitive to class balance)
                    class_balance = np.array([experiment_params['class_balance'],
                                              1.0 - experiment_params['class_balance']])
                    weighted = d[key] / (2.0 * class_balance[None, :, None])
                    diff_class = np.sum(weighted[:, 0, :], axis=-1) - np.sum(weighted[:, 1, :], axis=-1)
                    confidence = np.sum(np.max(weighted, axis=1), axis=-1) - np.abs(diff_class)
                    line_results['confidence_%s' % key].append(confidence)
                elif 'classification_acc' in key:
                    # standard
                    line_results[key].append(d[key])
                    # flipped
                    line_results['^%s' % key].append(np.maximum(d[key], 1 - d[key]))
                else:
                    line_results[key].append(d[key])

    # concatenate results
    line = {}
    n = None
    for key, value in line_results.items():
        if args.trim_epochs is not None:
            value = [v[:args.trim_epochs] for v in value]
        epochs = [len(v) for v in value]
        if len(np.unique(epochs)) > 1:
            print('%s: Number of epochs for %s do not match: %s' % (input_dir, key, str(np.unique(epochs))))
            value = [v[:min(epochs)] for v in value]
        line[key] = np.vstack(value)
        # assure that all results have the same shape (no lines or epochs missing)
        if n is None:
            n = line[key].shape
        else:
            assert n == line[key].shape

    if len(line_results) > 0:
        lines.append((input_dir, line))
        for key in line.keys():
            keys.add(key)

keys = list(sorted(keys))

# compute summary statistics
for input_dir, line in lines:
    for key in keys:
        # which epoch to use for the measurement
        for epoch in ('min', 'max', 'end'):
            if epoch == 'min':
                v = np.nanmin(line[key], axis=1)  # minimum over all epochs
            elif epoch == 'max':
                v = np.nanmax(line[key], axis=1)  # maximum over all epochs
            elif epoch == 'end':
                v = line[key][:, -1]           # final epoch
            else:
                assert False

            # summarize over all experiments
            line['%s[count@%s]' % (key, epoch)] = len(v)
            line['%s[min@%s]' % (key, epoch)] = np.nanmin(v)
            line['%s[mean@%s]' % (key, epoch)] = np.nanmean(v)
            line['%s[max@%s]' % (key, epoch)] = np.nanmax(v)
            count, bins = np.histogram(v, bins=np.linspace(0, 1, 11))
            line['%s[hist@%s]' % (key, epoch)] = ' '.join(str(c) for c in count / np.sum(count))


if args.text_output:
    for input_dir, line in lines:
        print(input_dir)
        for key in keys:
            print('  %s' % key, line[key].shape)
            epoch_strategy = {
                    'min': np.nanmin(line[key], axis=1),  # minimum over all epochs
                    'max': np.nanmax(line[key], axis=1),  # maximum over all epochs
                    'end': line[key][:, :-1],          # final epoch
                }
            for s, v in epoch_strategy.items():
                print('    %s: [%f, %f, %f]' % (s, np.nanmin(v), np.nanmean(v), np.nanmax(v)))


if args.html_table:
    print('<table>')
    print('<tr><td></td>', end='')
    for key in keys:
        print('<td colspan="9">%s</td>' % key, end='')
    print('</td></tr>')
    print('<tr><td>within-run</td>', end='')
    for key in keys:
        print('<td colspan="3">min</td>', end='')
        print('<td colspan="3">max</td>', end='')
        print('<td colspan="3">end</td>', end='')
    print('</td></tr>')
    print('<tr><td>within-run</td>', end='')
    for key in keys:
        for i in range(3):
            print('<td>min</td>', end='')
            print('<td>mean</td>', end='')
            print('<td>max</td>', end='')
    print('</td></tr>')
    for input_dir, line in lines:
        print('<tr><td>%s</td>' % input_dir, end='')
        for key in keys:
            # which epoch to use for the measurement
            for epoch in ('min', 'max', 'end'):
                # how to summarize over all experiments
                for summary in ('min', 'mean', 'max'):
                    k = '%s[%s@%s]' % (key, summary, epoch)
                    print('<td>%f</td>' % line[k], end='')
        print('</tr>')
    print('</table>')


if args.output is not None:
    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['experiment']
        for key in keys:
            # how to summarize over all experiments
            for summary in ('min', 'mean', 'max', 'hist'):
                # which epoch to use for the measurement
                for epoch in ('min', 'max', 'end'):
                    header.append('%s[%s@%s]' % (key, summary, epoch))
        writer.writerow(header)
        for input_dir, line in lines:
            columns = [re.sub(args.strip_dir_prefix, '', input_dir)]
            for key in keys:
                # how to summarize over all experiments
                for summary in ('min', 'mean', 'max', 'hist'):
                    # which epoch to use for the measurement
                    for epoch in ('min', 'max', 'end'):
                        k = '%s[%s@%s]' % (key, summary, epoch)
                        columns.append(line[k] if k in line else None)
            writer.writerow(columns)
