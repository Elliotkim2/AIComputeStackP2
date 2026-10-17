#!/usr/bin/env python3
"""H200 SXM attention roofline; default: Llama 3.1 70B, 128K context, BF16.

Install: python3 -m pip install matplotlib
Run: python3 plot_attention_rooflines.py
Outputs: attention_rooflines.png and attention_rooflines.pdf

Uses the attention-core arithmetic and fused-byte model in assignment section 2.4.
Points are analytical roof bounds, not measured GPU performance. This model
excludes projection/MLP operations and does not establish whole-model capacity.
"""
import argparse
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PEAK = 989500.0  # GFLOP/s: dense BF16/FP16 Tensor Core theoretical peak
BANDWIDTH = 4800.0  # GB/s: HBM bandwidth
SOURCE = 'https://www.nvidia.com/en-us/data-center/h200/'
MODEL_SOURCE = 'https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--query-heads', type=int, default=64)
    parser.add_argument('--kv-heads', type=int, default=8)
    parser.add_argument('--head-dim', type=int, default=128)
    parser.add_argument('--seqlens', type=int, nargs='+', default=[131072])
    parser.add_argument('--output', type=Path, default=Path('attention_rooflines'))
    args = parser.parse_args()
    if min(args.query_heads, args.kv_heads, args.head_dim, *args.seqlens) <= 0:
        parser.error('all attention dimensions must be positive')
    if args.query_heads % args.kv_heads:
        parser.error('query-heads must be divisible by kv-heads')

    s = 2  # BF16 bytes per element
    d = args.query_heads * args.head_dim
    dkv = args.kv_heads * args.head_dim
    ridge = PEAK / BANDWIDTH
    decode_i = 2 * d / (dkv * s)
    prefill_i = [2 * n * d / ((d + dkv) * s) for n in args.seqlens]
    xmin = min(decode_i, ridge, *prefill_i) / 4
    xmax = max(decode_i, ridge, *prefill_i) * 4
    xs = sorted(set([10 ** (math.log10(xmin) + j / 699 *
                          (math.log10(xmax) - math.log10(xmin))) for j in range(700)]
                    + [ridge]))
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.subplots_adjust(left=.10, right=.97, bottom=.28, top=.86)
    fig.suptitle('H200 SXM: fused attention roofline', fontsize=18, fontweight='bold', y=.97)
    fig.text(.5, .91, f'BF16 | query heads = {args.query_heads} | KV heads = '
             f'{args.kv_heads} | head dimension = {args.head_dim}', ha='center', fontsize=11)
    ax.plot(xs, [min(PEAK, x * BANDWIDTH) for x in xs], color='#238658', lw=2.5,
            label='H200 SXM: theoretical dense BF16 roof')
    ax.axvline(ridge, color='#238658', ls=':', lw=1.2)
    ax.scatter([ridge], [PEAK], marker='D', color='#238658', s=40, zorder=4)
    ax.annotate(f'Ridge = {ridge:.2f} FLOP/byte', (ridge, PEAK),
                xytext=(8, 13), textcoords='offset points', color='#238658', fontsize=10)
    for index, (n, intensity) in enumerate(zip(args.seqlens, prefill_i)):
        performance = min(PEAK, intensity * BANDWIDTH)
        ax.scatter([intensity], [performance], marker='^', color='#9345ad', s=110,
                   edgecolor='white', zorder=5,
                   label='Fused prefill bound' if index == 0 else None)
        ax.annotate(f'Prefill: n={n:,}\nI={intensity:,.2f}', (intensity, performance),
                    xytext=(-8, -38), textcoords='offset points', ha='right',
                    fontsize=10, color='#73318a')
        print(f'Prefill n={n}: intensity={intensity:.6f} FLOP/byte; '
              f'roof={performance / 1000:.3f} TFLOP/s')
    decode_p = min(PEAK, decode_i * BANDWIDTH)
    ax.scatter([decode_i], [decode_p], marker='X', color='#d05a22', s=110,
               edgecolor='white', zorder=5, label='Fused decode bound')
    ax.annotate(f'Decode: I={decode_i:g}\n{decode_p / 1000:g} TFLOP/s bound',
                (decode_i, decode_p), xytext=(10, -34), textcoords='offset points',
                fontsize=10, color='#b34c1c')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(min(PEAK, xmin * BANDWIDTH) / 1.7, PEAK * 4)
    ax.set_xlabel('Arithmetic intensity (FLOP/byte)', fontsize=12)
    ax.set_ylabel('Performance (GFLOP/s)', fontsize=12)
    ax.grid(which='major', alpha=.23)
    ax.grid(which='minor', alpha=.07)
    ax.legend(loc='lower right', fontsize=10, framealpha=.95)
    fig.text(.10, .13,
             f'Decode intensity is {ridge / decode_i:.2f} times below the ridge. '
             'Attention points are analytical bounds, not measurements.\n'
             'Default architecture: Llama 3.1 70B at 131,072 tokens. '
             'Fused-byte model from assignment section 2.4.\n'
             'The attention-core model excludes projection/MLP costs and whole-model memory capacity.',
             fontsize=9, linespacing=1.6)
    fig.text(.10, .045, 'H200 specifications: ' + SOURCE + '\n'
             'Dense peak = 1,979 / 2 = 989.5 TFLOP/s; bandwidth = 4.8 TB/s.',
             fontsize=8, color='#444444', linespacing=1.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        path = Path(str(args.output) + '.' + ext)
        fig.savefig(path, dpi=220, facecolor='white')
        print(f'Saved {path}')
    plt.close(fig)
    print(f'Decode: intensity={decode_i:.6f} FLOP/byte; ridge/intensity={ridge / decode_i:.6f}')


if __name__ == '__main__':
    main()
