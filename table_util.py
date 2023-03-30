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

import random


sparklines_out = []


def random_string_generator(str_size, allowed_chars):
    return ''.join(random.choice(allowed_chars) for x in range(str_size))


def sparkline(vals, context):
    sparkline_id = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for x in range(5))

    vals = vals.split(' ')
    assert len(vals) == 10
    c = """
      \\begin{sparkline}{5}
        \\definecolor{sparkbottomlinecolor}{gray}{0.9}
        \\definecolor{sparkspikecolor}{gray}{0.5}
        \\setlength\\sparkbottomlinethickness{0.5pt}
        \\sparkbottomline
        \\setlength\\sparkspikewidth{0.45ex}"""
    for x, v in zip(range(len(vals)), vals):
        c += ' \\sparkspike %f %s' % ((x / 10), v)
    c += """
      \\end{sparkline}"""

    s = "\\newcommand{\\sP%s}{%s}\n" % (sparkline_id, c)

    sparklines_out.append(s)
    return "\\sP%s" % sparkline_id


def render_table(begin_table, end_table, experiments, columns, rows):
    out = []

    out.append(begin_table + '\n')

    max_row_title_len = max(len(row[0]) for row in rows if not isinstance(row, str))
    row_title_format = '  %' + str(max_row_title_len) + 's '

    for row_idx, row in enumerate(rows):
        if isinstance(row, str):
            out.append('  %s' % row)
        elif row[1] is None:
            row_title = '\\multicolumn{%d}{l}{%s}' % (len(columns) + 1, row[0])
            if row_idx > 0:
                row_title = '\\addlinespace[0.6em]  ' + row_title
            out.append(row_title_format % row_title + ' \\\\\n')
        else:
            row_title, experiment = row
            out.append(row_title_format % row_title)
            for column in columns:
                out.append(' & ')
                for key in list(column):
                    if 'hist' in key:
                        if experiment in experiments and key in experiments[experiment]:
                            out.append(sparkline(experiments[experiment][key], context=key))
                    else:
                        if experiment in experiments and key in experiments[experiment]:
                            value = 100 * float(experiments[experiment][key])
                            value_str = '%0.1f' % value
                            if len(value_str) < len('000.0'):
                                value_str = '\\hphantom{' + ('0' * (len('000.0') - len(value_str))) + '}' + value_str
                            out.append('%6s  ' % value_str)
                        else:
                            out.append('%6s  ' % '')
            out.append(' \\\\\n')

    out.append(end_table)

    print('{')
    print(''.join(sparklines_out))
    print(''.join(out))
    print('}')
