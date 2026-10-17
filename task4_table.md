# T4.1: Measured machine comparison

| Machine | Compute (GFLOP/s) | Bandwidth (GB/s) | Ridge (FLOP/byte) |
| --- | ---: | ---: | ---: |
| Machine 1 (local) | 110.896 | 41.651 | 2.662 |
| Machine 2 (CRC) | 531.870 | 10.166 | 52.319 |
| Ratio (CRC / local) | 4.796 | 0.244 | 19.650 |

The ratio row is dimensionless. Ridge = compute / bandwidth. All calculations use the full-precision CSV values.

## Measurement sources and recorded provenance

### Machine 1 (local)

Source CSV: p2_group07_roofline.csv

```json
{
  "machine": "x86_64",
  "platform": "Linux-5.14.0-687.56.1.el9_8.x86_64-x86_64-with-glibc2.34",
  "python": "3.13.2",
  "numpy": "2.5.4",
  "cpu_count": "64",
  "OMP_NUM_THREADS": "1",
  "dtype": "fp32",
  "blas": "scipy-openblas"
}
```

### Machine 2 (CRC)

Source CSV: p2_crc_roofline.csv

```json
{
  "machine": "x86_64",
  "platform": "Linux-4.18.0-553.157.1.el8_10.x86_64-x86_64-with-glibc2.28",
  "python": "3.12.4",
  "numpy": "2.1.2",
  "cpu_count": "32",
  "OMP_NUM_THREADS": "unset",
  "dtype": "fp32",
  "blas": "scipy-openblas"
}
```

These metadata blocks are extracted from the CSV, not copied from the original console PROVENANCE lines. Include the original console blocks in the report as required by the assignment.

cpu_count is the recorded logical CPU count; it does not verify the number of BLAS threads actually used. OMP_NUM_THREADS=unset means the variable was not explicitly set, not that only one thread was used.

If these calibration ceilings are used with sweeps from different runs, state that choice and use the selected ceilings consistently in the combined roofline and recomputed percentages.
