#!/usr/bin/env python3
"""Make the T4.1 comparison table from one existing CSV per machine.

No benchmarks or third-party packages required. Example:
  python3 task4_table.py --local p2_group07_roofline.csv \
      --crc p2_crc_skinny_roofline.csv

To use a separate machine-only calibration, pass its CSV instead (for example
--crc p2_crc_roofline.csv). Never mix a compute ceiling from one run with a
bandwidth ceiling from another. Ridges are calculated from unrounded values.
"""

import argparse
import csv
import json
import math
from pathlib import Path

PROVENANCE = (
    'machine', 'platform', 'python', 'numpy', 'cpu_count',
    'OMP_NUM_THREADS', 'dtype', 'blas',
)


def load_machine(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'{path}: empty CSV')
    pairs = set()
    for line, row in enumerate(rows, 2):
        try:
            peak = float(row['machine_peak_gflops'])
            bandwidth = float(row['machine_bandwidth_gbs'])
            if any(not math.isfinite(v) or v <= 0 for v in (peak, bandwidth)):
                raise ValueError('ceilings must be finite and positive')
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'{path}, line {line}: invalid ceilings ({exc})') from exc
        pairs.add((peak, bandwidth))
    if len(pairs) != 1:
        raise ValueError(f'{path}: multiple ceiling pairs; use one run per CSV')
    peak, bandwidth = pairs.pop()
    provenance = {}
    for key in PROVENANCE:
        values = {r.get(key, '') for r in rows}
        if len(values) != 1:
            raise ValueError(f'{path}: inconsistent provenance field {key}')
        provenance[key] = values.pop()
    return (peak, bandwidth, peak / bandwidth), provenance


def md_row(row):
    return '| ' + ' | '.join(str(v).replace('|', '\\|') for v in row) + ' |'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--local', type=Path, required=True, help='Machine 1 CSV')
    parser.add_argument('--crc', type=Path, required=True, help='Machine 2 CSV')
    parser.add_argument('--local-label', default='Machine 1 (local)')
    parser.add_argument('--crc-label', default='Machine 2 (CRC)')
    parser.add_argument('--output-prefix', type=Path, default=Path('task4_table'))
    args = parser.parse_args()
    try:
        local, local_prov = load_machine(args.local)
        crc, crc_prov = load_machine(args.crc)
        if local_prov['dtype'] != crc_prov['dtype']:
            raise ValueError('Machine precisions differ; compare matching dtypes')
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f'Error: {exc}\n')
    ratios = tuple(second / first for first, second in zip(local, crc))
    headers = ['Machine', 'Compute (GFLOP/s)', 'Bandwidth (GB/s)', 'Ridge (FLOP/byte)']
    data = [[args.local_label, *local], [args.crc_label, *crc],
            ['Ratio (CRC / local)', *ratios]]
    # CSV keeps full numeric precision; Markdown rounds only for display.
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(str(prefix) + '.csv')
    md_path = Path(str(prefix) + '.md')
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(data)
    lines = ['# T4.1: Measured machine comparison', '', md_row(headers),
             md_row(['---', '---:', '---:', '---:'])]
    for label, *numbers in data:
        lines.append(md_row([label, *(f'{value:.3f}' for value in numbers)]))
    lines.extend(['', 'The ratio row is dimensionless. Ridge = compute / bandwidth. '
                  'All calculations use the full-precision CSV values.', '',
                  '## Measurement sources and recorded provenance', ''])
    for label, path, provenance in [(args.local_label, args.local, local_prov),
                                    (args.crc_label, args.crc, crc_prov)]:
        lines.extend([f'### {label}', '', f'Source CSV: {path.name}', '',
                      '```json', json.dumps(provenance, indent=2), '```', ''])
    lines.extend([
        'These metadata blocks are extracted from the CSV, not copied from the '
        'original console PROVENANCE lines. Include the original console blocks '
        'in the report as required by the assignment.', '',
        'cpu_count is the recorded logical CPU count; it does not verify the '
        'number of BLAS threads actually used. OMP_NUM_THREADS=unset means the '
        'variable was not explicitly set, not that only one thread was used.', '',
        'If these calibration ceilings are used with sweeps from different runs, '
        'state that choice and use the selected ceilings consistently in the '
        'combined roofline and recomputed percentages.',
    ])
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:9]))
    print(f'\nSaved {csv_path} and {md_path}')


if __name__ == '__main__':
    main()
