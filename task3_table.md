# Task 3: GEMM classification table

Computed from existing measurements; no benchmarks were run.

Roof = min(compute ceiling, intensity * bandwidth ceiling). Percent of roof = 100 * achieved / roof. Cold-cache intensities are taken directly from the input CSVs. Percentages above 100 are retained.

Classification heuristic: at least 80% of the roof is provisionally classified by the lower roof; points below that are labeled Neither / underutilized. This threshold is an analysis choice, not an assignment rule. Low utilization does not identify its cause, and proximity to a roof is evidence rather than proof of a bottleneck.

## Measured ceilings used

| Source | Compute (GFLOP/s) | Bandwidth (GB/s) | Ridge (FLOP/byte) |
| --- | ---: | ---: | ---: |
| p2_group07_roofline.csv | 110.896 | 41.651 | 2.6625 |
| p2_group07_square_roofline.csv | 110.167 | 22.335 | 4.9325 |

**Note:** Inputs use different measured ceiling pairs. Each row keeps its own run ceilings; do not describe these as one common roof without reconciling the measurements.

## Configurations

| Source | Family | M | N | K | Dtype | Intensity (FLOP/byte) | Achieved (GFLOP/s) | Roof (GFLOP/s) | Percent of roof | Roofline prediction | Classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p2_group07_roofline.csv | Skinny | 1024 | 1024 | 1 | fp32 | 0.499 | 1.624 | 20.785 | 7.81% | Bandwidth | Neither / underutilized |
| p2_group07_roofline.csv | Skinny | 1024 | 1024 | 2 | fp32 | 0.996 | 8.507 | 41.489 | 20.50% | Bandwidth | Neither / underutilized |
| p2_group07_roofline.csv | Skinny | 1024 | 1024 | 4 | fp32 | 1.984 | 17.272 | 82.656 | 20.90% | Bandwidth | Neither / underutilized |
| p2_group07_roofline.csv | Skinny | 1024 | 1024 | 8 | fp32 | 3.938 | 31.759 | 110.896 | 28.64% | Compute | Neither / underutilized |
| p2_group07_roofline.csv | Skinny | 1024 | 1024 | 16 | fp32 | 7.758 | 64.870 | 110.896 | 58.50% | Compute | Neither / underutilized |
| p2_group07_roofline.csv | Skinny | 1024 | 1024 | 32 | fp32 | 15.059 | 84.926 | 110.896 | 76.58% | Compute | Neither / underutilized |
| p2_group07_roofline.csv | Skinny | 1024 | 1024 | 64 | fp32 | 28.444 | 97.716 | 110.896 | 88.12% | Compute | Compute-bound |
| p2_group07_roofline.csv | Skinny | 1024 | 1024 | 128 | fp32 | 51.200 | 107.145 | 110.896 | 96.62% | Compute | Compute-bound |
| p2_group07_roofline.csv | Skinny | 1024 | 1024 | 256 | fp32 | 85.333 | 110.702 | 110.896 | 99.83% | Compute | Compute-bound |
| p2_group07_roofline.csv | Skinny | 1024 | 1024 | 512 | fp32 | 128.000 | 110.548 | 110.896 | 99.69% | Compute | Compute-bound |
| p2_group07_roofline.csv | Square | 1024 | 1024 | 1024 | fp32 | 170.667 | 111.003 | 110.896 | 100.10% | Compute | Compute-bound |
| p2_group07_roofline.csv | Skinny | 1024 | 1024 | 2048 | fp32 | 204.800 | 110.736 | 110.896 | 99.86% | Compute | Compute-bound |
| p2_group07_square_roofline.csv | Square | 32 | 32 | 32 | fp32 | 5.333 | 26.267 | 110.167 | 23.84% | Compute | Neither / underutilized |
| p2_group07_square_roofline.csv | Square | 64 | 64 | 64 | fp32 | 10.667 | 59.403 | 110.167 | 53.92% | Compute | Neither / underutilized |
| p2_group07_square_roofline.csv | Square | 128 | 128 | 128 | fp32 | 21.333 | 66.854 | 110.167 | 60.68% | Compute | Neither / underutilized |
| p2_group07_square_roofline.csv | Square | 192 | 192 | 192 | fp32 | 32.000 | 105.181 | 110.167 | 95.47% | Compute | Compute-bound |
| p2_group07_square_roofline.csv | Square | 256 | 256 | 256 | fp32 | 42.667 | 106.901 | 110.167 | 97.04% | Compute | Compute-bound |
| p2_group07_square_roofline.csv | Square | 384 | 384 | 384 | fp32 | 64.000 | 108.041 | 110.167 | 98.07% | Compute | Compute-bound |
| p2_group07_square_roofline.csv | Square | 512 | 512 | 512 | fp32 | 85.333 | 109.760 | 110.167 | 99.63% | Compute | Compute-bound |
| p2_group07_square_roofline.csv | Square | 768 | 768 | 768 | fp32 | 128.000 | 110.377 | 110.167 | 100.19% | Compute | Compute-bound |
| p2_group07_square_roofline.csv | Square | 1024 | 1024 | 1024 | fp32 | 170.667 | 109.401 | 110.167 | 99.30% | Compute | Compute-bound |
| p2_group07_square_roofline.csv | Square | 1536 | 1536 | 1536 | fp32 | 256.000 | 110.276 | 110.167 | 100.10% | Compute | Compute-bound |
