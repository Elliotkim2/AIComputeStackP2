#!/usr/bin/env python3
"""Generate Task 3 tables from existing p2_roofline_lab CSVs (no benchmarks).

Python standard library only. Example:
  python task3_table.py p2_group07_roofline.csv p2_group07_square_roofline.csv

Uses each row's measured compute and bandwidth ceilings. Classifications are
provisional: below --near-roof-percent is 'Neither / underutilized'; otherwise
the lower roof determines the classification. The default 80% is an explicit
analysis heuristic, NOT a requirement of the assignment or proof of a bottleneck.
Inspect the results and explain outliers using evidence in your report.
"""

import argparse
import csv
import math
from pathlib import Path
import sys


FIELDS = [
    'Source', 'Family', 'M', 'N', 'K', 'Dtype', 'Intensity (FLOP/byte)',
    'Achieved (GFLOP/s)', 'Roof (GFLOP/s)', 'Percent of roof',
    'Roofline prediction', 'Classification',
]


def positive(row, key):
    value = float(row[key])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{key} must be finite and positive')
    return value


def collect(paths, threshold):
    table, ceilings = [], []
    for path in paths:
        count = 0
        pairs = set()
        with path.open(newline='', encoding='utf-8-sig') as stream:
            for line, row in enumerate(csv.DictReader(stream), start=2):
                if row.get('probe', '').strip() != 'gemm':
                    continue  # Compute/bandwidth probes calibrate roofs, not sweep points.
                try:
                    m, n, k = (int(row[key]) for key in ('M', 'N', 'K'))
                    if min(m, n, k) <= 0:
                        raise ValueError('dimensions must be positive')
                    intensity = positive(row, 'arithmetic_intensity')
                    achieved = positive(row, 'gflops')
                    peak = positive(row, 'machine_peak_gflops')
                    bandwidth = positive(row, 'machine_bandwidth_gbs')
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f'{path}, line {line}: {exc}') from exc
                roof = min(peak, intensity * bandwidth)
                percent = 100 * achieved / roof
                prediction = ('Bandwidth' if intensity < peak / bandwidth else 'Compute')
                classification = (prediction + '-bound' if percent >= threshold
                                  else 'Neither / underutilized')
                family = row.get('family', '').strip()
                if not family:
                    family = ('Square' if m == n == k else
                              'Skinny' if m == n == 1024 else 'Other')
                table.append([
                    path.name, family, str(m), str(n), str(k), row.get('dtype', ''),
                    f'{intensity:.3f}', f'{achieved:.3f}', f'{roof:.3f}',
                    f'{percent:.2f}%', prediction, classification,
                ])
                pairs.add((peak, bandwidth))
                count += 1
        if not count:
            raise ValueError(f'{path}: no gemm rows found')
        for peak, bandwidth in sorted(pairs):
            ceilings.append((path.name, peak, bandwidth))
    return table, ceilings


def markdown_row(values):
    return '| ' + ' | '.join(str(v).replace('|', '\\|').replace('\n', ' ')
                             for v in values) + ' |'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv_files', nargs='+', type=Path)
    parser.add_argument('--output-prefix', default='task3_table', type=Path)
    parser.add_argument('--near-roof-percent', type=float, default=80.0,
                        help='Provisional classification threshold; default: 80 (heuristic)')
    args = parser.parse_args()
    if not 0 < args.near_roof_percent <= 100:
        parser.error('--near-roof-percent must be greater than 0 and at most 100')
    try:
        table, ceilings = collect(args.csv_files, args.near_roof_percent)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f'Error: {exc}\n')

    notes = [
        '# Task 3: GEMM classification table', '',
        'Computed from existing measurements; no benchmarks were run.', '',
        'Roof = min(compute ceiling, intensity * bandwidth ceiling). '
        'Percent of roof = 100 * achieved / roof. Cold-cache intensities are '
        'taken directly from the input CSVs. Percentages above 100 are retained.', '',
        f'Classification heuristic: at least {args.near_roof_percent:g}% of the roof '
        'is provisionally classified by the lower roof; points below that are '
        'labeled Neither / underutilized. This threshold is an analysis choice, '
        'not an assignment rule. Low utilization does not identify its cause, '
        'and proximity to a roof is evidence rather than proof of a bottleneck.', '',
        '## Measured ceilings used', '',
        '| Source | Compute (GFLOP/s) | Bandwidth (GB/s) | Ridge (FLOP/byte) |',
        '| --- | ---: | ---: | ---: |',
    ]
    for source, peak, bandwidth in ceilings:
        notes.append(markdown_row([source, f'{peak:.3f}', f'{bandwidth:.3f}',
                                   f'{peak / bandwidth:.4f}']))
    if len({(p, b) for _, p, b in ceilings}) > 1:
        warning = ('Inputs use different measured ceiling pairs. Each row keeps '
                   'its own run ceilings; do not describe these as one common roof '
                   'without reconciling the measurements.')
        notes.extend(['', '**Note:** ' + warning])
        print('Note: ' + warning, file=sys.stderr)
    notes.extend(['', '## Configurations', '', markdown_row(FIELDS),
                  markdown_row(['---'] * len(FIELDS))])
    notes.extend(markdown_row(row) for row in table)
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = Path(str(prefix) + '.csv'), Path(str(prefix) + '.md')
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        writer.writerows(table)
    md_path.write_text('\n'.join(notes) + '\n', encoding='utf-8')
    print(f'Generated {len(table)} GEMM configurations.')
    print(f'CSV: {csv_path}\nMarkdown: {md_path}')


if __name__ == '__main__':
    main()
