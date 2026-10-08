#!/usr/bin/env python3
"""
p2_attention_intensity.py -- Project 2, Task 4: where does an attention layer
land on the roofline?

CSE 40702, The AI Compute Stack.

Computes the arithmetic intensity of one attention layer ANALYTICALLY, for
four cases that land in very different places:

    prefill, fused   (FlashAttention-style, score matrix never materialized)
    prefill, naive   (score matrix written to and read from HBM)
    decode,  fused
    decode,  naive

    python p2_attention_intensity.py --model Qwen/Qwen3-0.6B --seqlens 512 2048 8192
    python p2_attention_intensity.py --manual 32 32 8 128 --seqlens 1024 --ridge 295

Pass --ridge with the ridge point you measured in Task 1 and the script will
classify each case for you.  Do not put that classification in your report
without deriving at least one of these four by hand first; you will be asked
to reproduce it on a whiteboard.

THE ARITHMETIC, SO YOU CAN CHECK IT
-----------------------------------
Let n = sequence length, h_q = query heads, h_kv = key/value heads,
d_h = head dim, d = h_q * d_h, d_kv = h_kv * d_h, s = bytes per element.

PREFILL (process n tokens at once), per layer:
  FLOPs:  S = QK^T  -> 2 * n * n * d
          O = AV    -> 2 * n * n * d
          total     -> 4 n^2 d
  Bytes, fused:  read Q (nd), K (n d_kv), V (n d_kv), write O (nd)
                 -> (2nd + 2n d_kv) * s
  Bytes, naive:  the same, PLUS write and read the score matrix,
                 2 * n^2 * h_q * s

DECODE (one new token against a cache of length n), per layer:
  FLOPs:  2 n d  +  2 n d   ->  4 n d
  Bytes:  read the whole KV cache, 2 n d_kv s, plus q and o, which are O(d)
          and negligible for large n

The two results worth carrying into the defense:

  1. NAIVE PREFILL intensity is INDEPENDENT OF n.  It converges to 2 d_h / s.
     No matter how long the sequence, materializing the score matrix pins you
     at a fixed, low intensity.  Fused prefill intensity grows like n/s.
     That difference is the entire argument for FlashAttention, and it is a
     roofline argument, not a memory-capacity one.

  2. DECODE intensity is a CONSTANT, 2 * (h_q / h_kv) / s, independent of n.
     For multi-head attention at fp16 that is 1 FLOP per byte. The ridge point
     of a modern accelerator is 150-300. Decode attention is therefore
     bandwidth-bound by two orders of magnitude, and no amount of batching
     fixes it, because every sequence in the batch has its own cache.
"""

import argparse
import sys


def facts_from_hf(model_id):
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(model_id)
    L = getattr(c, "num_hidden_layers", None) or getattr(c, "n_layer")
    hq = getattr(c, "num_attention_heads", None) or getattr(c, "n_head")
    hid = getattr(c, "hidden_size", None) or getattr(c, "n_embd")
    hkv = getattr(c, "num_key_value_heads", None) or hq
    dh = getattr(c, "head_dim", None) or (hid // hq)
    return L, hq, hkv, dh


def intensity(n, hq, hkv, dh, s):
    d, dkv = hq * dh, hkv * dh

    pre_flops = 4.0 * n * n * d
    pre_bytes_fused = (2.0 * n * d + 2.0 * n * dkv) * s
    pre_bytes_naive = pre_bytes_fused + 2.0 * n * n * hq * s

    dec_flops = 4.0 * n * d
    dec_bytes_fused = (2.0 * n * dkv + 2.0 * d) * s
    dec_bytes_naive = dec_bytes_fused + 2.0 * n * hq * s

    return {
        "prefill_fused": (pre_flops, pre_bytes_fused, pre_flops / pre_bytes_fused),
        "prefill_naive": (pre_flops, pre_bytes_naive, pre_flops / pre_bytes_naive),
        "decode_fused":  (dec_flops, dec_bytes_fused, dec_flops / dec_bytes_fused),
        "decode_naive":  (dec_flops, dec_bytes_naive, dec_flops / dec_bytes_naive),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="HuggingFace model id to read the config from")
    ap.add_argument("--manual", nargs=4, type=int,
                    metavar=("LAYERS", "H_Q", "H_KV", "D_HEAD"),
                    help="give the architecture directly instead")
    ap.add_argument("--seqlens", type=int, nargs="+",
                    default=[128, 512, 2048, 8192, 32768])
    ap.add_argument("--bytes", type=int, default=2, help="bytes per element")
    ap.add_argument("--ridge", type=float,
                    help="your measured ridge point, FLOP/byte, from Task 1")
    args = ap.parse_args()

    if args.manual:
        L, hq, hkv, dh = args.manual
        src = "manual"
    elif args.model:
        L, hq, hkv, dh = facts_from_hf(args.model)
        src = args.model
    else:
        ap.error("give --model or --manual")

    print(f"ARCHITECTURE  source={src}  layers={L}  h_q={hq}  h_kv={hkv}  "
          f"d_head={dh}  d_model_attn={hq*dh}  bytes={args.bytes}")
    print(f"              GQA ratio h_q/h_kv = {hq/hkv:.0f}")

    print(f"\n{'n':>8} {'case':>15} {'GFLOP':>10} {'MB moved':>10} "
          f"{'I (F/B)':>10}" + ("  bound-by" if args.ridge else ""))
    for n in args.seqlens:
        for case, (f, b, i) in intensity(n, hq, hkv, dh, args.bytes).items():
            tag = ""
            if args.ridge:
                tag = "  compute" if i >= args.ridge else "  bandwidth"
            print(f"{n:>8} {case:>15} {f/1e9:>10.2f} {b/2**20:>10.2f} "
                  f"{i:>10.1f}{tag}")
        print()

    dec_const = 2.0 * (hq / hkv) / args.bytes
    naive_pre_limit = 2.0 * dh / args.bytes
    print(f"Closed forms for this architecture:")
    print(f"  decode intensity, any n        = 2*(h_q/h_kv)/s = {dec_const:.2f} FLOP/byte")
    print(f"  naive prefill limit, large n   = 2*d_h/s        = {naive_pre_limit:.2f} FLOP/byte")
    print(f"  fused prefill grows like n/s; it crosses the naive limit near "
          f"n = {int(naive_pre_limit*args.bytes)}")
    if args.ridge:
        n_cross = args.ridge * args.bytes
        print(f"  fused prefill reaches your ridge ({args.ridge:g}) near "
              f"n = {n_cross:.0f}")
        print(f"\n  Decode sits {args.ridge/dec_const:.0f}x below the ridge. "
              f"State that number in the memo.")

    print("\nFor the report: put at least the fused prefill and fused decode "
          "points on your roofline, at a sequence length you justify, and say "
          "why they land where they do.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
