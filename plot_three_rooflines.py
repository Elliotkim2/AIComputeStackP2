#!/usr/bin/env python3
"""T4.6: plot two measured CPU rooflines and a theoretical H200 SXM roof.

Requires matplotlib: python3 -m pip install matplotlib
Example (selected CRC machine-only calibration CSV):
  python3 plot_three_rooflines.py \
    --local p2_group07_roofline.csv p2_group07_square_roofline.csv \
    --crc p2_crc_skinny_roofline.csv p2_crc_square_roofline.csv \
    --crc-calibration p2_crc_roofline.csv

If the machine-only CSV is unavailable, explicitly supply its displayed values:
  ... --crc-ceilings 531.870 10.166

CSV points retain their recorded arithmetic_intensity and gflops coordinates.
One ceiling pair is selected per machine; no averaging across separate runs.
Without an explicit calibration/ceiling pair, the first sweep CSV supplies it.
"""

import argparse
import csv
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

H200_PEAK = 989500.0  # GFLOP/s, theoretical dense BF16/FP16 Tensor Core
H200_BW = 4800.0     # GB/s, HBM3e bandwidth (not interconnect bandwidth)
SOURCE = 'https://www.nvidia.com/en-us/data-center/h200/'


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError('expected finite positive number')
    return number


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'{path}: empty CSV')
    return rows


def ceilings(rows):
    values = {(positive(r['machine_peak_gflops']),
               positive(r['machine_bandwidth_gbs'])) for r in rows}
    if len(values) != 1:
        raise ValueError('CSV must contain exactly one calibration pair')
    return next(iter(values))


def load_machine(paths, calibration, explicit):
    datasets = [(p, read_csv(p)) for p in paths]
    if explicit:
        selected = tuple(positive(v) for v in explicit)
        calibration_source = 'explicit ceiling values'
    elif calibration:
        selected = ceilings(read_csv(calibration))
        calibration_source = str(calibration)
    else:
        selected = ceilings(datasets[0][1])
        calibration_source = str(paths[0])
    points, anchors, anchor_seen = [], [], set()
    for path, rows in datasets:
        if {r.get('dtype') for r in rows} != {'fp32'}:
            raise ValueError(f'{path}: expected FP32 CPU measurements')
        if ceilings(rows) != selected:
            print(f'Note: {path.name} has its own calibration; plotting its points '
                  f'unchanged against the selected roof ({calibration_source}).',
                  file=sys.stderr)
        for r in rows:
            probe = r.get('probe', '')
            if probe not in ('gemm', 'anchor'):
                continue
            x, y = positive(r['arithmetic_intensity']), positive(r['gflops'])
            if probe == 'gemm':
                m, n, k = (int(r[key]) for key in ('M', 'N', 'K'))
                family = 'Square' if m == n == k else 'Skinny' if m == n == 1024 else 'Other'
                points.append((x, y, family))
            else:
                # Each sweep repeats the same anchor probes; show the first set.
                key = r.get('kernel') or str(x)
                if key not in anchor_seen:
                    anchors.append((x, y))
                    anchor_seen.add(key)
    if not points:
        raise ValueError('no GEMM points found')
    return selected, points, anchors


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--local', nargs='+', type=Path, required=True)
    parser.add_argument('--crc', nargs='+', type=Path, required=True)
    for name in ('local', 'crc'):
        group = parser.add_mutually_exclusive_group()
        group.add_argument(f'--{name}-calibration', type=Path)
        group.add_argument(f'--{name}-ceilings', nargs=2, type=float,
                           metavar=('GFLOPS', 'GB_PER_S'))
    parser.add_argument('--output', type=Path, default=Path('three_rooflines'))
    args = parser.parse_args()
    try:
        local = load_machine(args.local, args.local_calibration, args.local_ceilings)
        crc = load_machine(args.crc, args.crc_calibration, args.crc_ceilings)
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        parser.exit(1, f'Error: {exc}\n')

    fig, ax = plt.subplots(figsize=(12, 7.8))
    fig.subplots_adjust(left=.09, right=.97, bottom=.30, top=.88)
    fig.suptitle('Three rooflines: measured CPUs and theoretical H200 SXM',
                 fontsize=17, fontweight='bold', y=.97)
    colors = ['#2563a6', '#d05a22', '#238658']
    machines = [('Local CPU', local[0]), ('CRC CPU', crc[0]),
                ('H200 SXM', (H200_PEAK, H200_BW))]
    all_points = local[1] + crc[1]
    all_anchors = local[2] + crc[2]
    xmin = min(x for x, *_ in all_points + all_anchors) / 2
    xmax = max(1000, max(x for x, *_ in all_points) * 2)
    logmin, logmax = math.log10(xmin), math.log10(xmax)
    base_grid = [10 ** (logmin + (logmax - logmin) * j / 699) for j in range(700)]
    roof_handles = []
    for (name, (peak, bandwidth)), color in zip(machines, colors):
        ridge = peak / bandwidth
        grid = sorted(set(base_grid + [ridge]))  # Include the exact corner.
        line, = ax.plot(grid, [min(peak, x * bandwidth) for x in grid],
                        color=color, linewidth=2,
                        label=f'{name}: {peak:,.1f} GFLOP/s; {bandwidth:,.2f} GB/s')
        roof_handles.append(line)
        ax.axvline(ridge, color=color, linewidth=1, linestyle=':', alpha=.65)
        ax.plot(ridge, peak, marker='D', markersize=5, color=color)
        ax.annotate(f'I* = {ridge:.2f}', (ridge, peak), xytext=(7, 9),
                    textcoords='offset points', color=color, fontsize=10,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=1))
    for machine, color in zip((local, crc), colors):
        for family, marker in [('Skinny', 'o'), ('Square', 's'), ('Other', '^')]:
            pts = [(x, y) for x, y, fam in machine[1] if fam == family]
            if pts:
                ax.scatter(*zip(*pts), marker=marker, s=38, color=color,
                           edgecolors='white', linewidths=.5, zorder=4)
        if machine[2]:
            ax.scatter(*zip(*machine[2]), marker='*', s=105, color=color,
                       edgecolors='white', linewidths=.5, zorder=4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(min(y for _, y, *_ in all_points + all_anchors) / 2, H200_PEAK * 3)
    ax.set_xlabel('Arithmetic intensity (FLOP/byte; recorded CPU cold-cache model)', fontsize=11)
    ax.set_ylabel('Performance (GFLOP/s)', fontsize=11)
    ax.grid(which='major', alpha=.23)
    ax.grid(which='minor', alpha=.07)
    roof_legend = ax.legend(handles=roof_handles, loc='upper left', fontsize=9,
                            framealpha=.95)
    ax.add_artist(roof_legend)
    symbols = [Line2D([], [], color=color, marker=marker, linestyle='None',
                      label=f'{machine_name}: {label}', markersize=7)
               for machine_name, color in [('Local', colors[0]), ('CRC', colors[1])]
               for label, marker in [('skinny GEMM', 'o'), ('square GEMM', 's'),
                                      ('streaming anchor', '*')]]
    ax.legend(handles=symbols, loc='lower right', fontsize=9)
    fig.text(.09, .135,
             'CPU points are measured FP32 and retain their original coordinates. CPU roofs use selected sustained calibrations.\n'
             'H200 roof: one GPU, theoretical dense BF16/FP16 Tensor Core peak; no GPU performance was measured.\n'
             'Precision changes arithmetic intensity: these CPU points are not BF16 performance predictions.',
             fontsize=9, linespacing=1.6)
    fig.text(.09, .045, 'H200 source: ' + SOURCE + '\n'
             '1,979 sparse TFLOP/s / 2 = 989.5 dense TFLOP/s; HBM bandwidth = 4.8 TB/s.',
             fontsize=8, color='#444444', linespacing=1.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        path = Path(str(args.output) + '.' + extension)
        fig.savefig(path, dpi=220, facecolor='white')
        print(f'Saved {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
