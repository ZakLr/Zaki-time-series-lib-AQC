# ESN Tuning Experiment
**Run ID**: 20260525_202715  

## GSOD Results (Top 10)

| Rank | Config | h=1 RMSE | Skill | VPT | FSDH | Time(s) |
|------|--------|----------|-------|-----|------|---------|
| 1 | ESN500_rad0.5_leak1.0_inp0.1_ridge1.0 | 0.6591 | -0.0785 | 2 | 0 | 9.0 |
| 2 | ESN500_rad0.9_leak1.0_inp0.1_ridge1.0 | 0.7387 | -0.2088 | 2 | 0 | 9.3 |
| 3 | ESN500_rad0.9_leak1.0_inp0.1_ridge0.0001 | 1.0770 | -0.7625 | 1 | 0 | 9.3 |
| 4 | ESN500_rad1.5_leak1.0_inp0.1_ridge0.0001 | 1.2559 | -1.0553 | 1 | 0 | 9.2 |
| 5 | ESN500_rad1.5_leak1.0_inp0.1_ridge1.0 | 1.2564 | -1.0561 | 1 | 0 | 9.2 |
| 6 | ESN500_rad1.5_leak0.3_inp0.1_ridge1.0 | 1.3344 | -1.1837 | 1 | 0 | 9.5 |
| 7 | ESN500_rad1.5_leak0.3_inp0.1_ridge0.0001 | 1.3410 | -1.1945 | 1 | 0 | 9.6 |
| 8 | ESN500_rad0.9_leak0.3_inp0.1_ridge1.0 | 1.3968 | -1.2859 | 1 | 0 | 9.2 |
| 9 | ESN500_rad0.5_leak0.3_inp0.1_ridge1.0 | 1.5835 | -1.5914 | 1 | 0 | 9.2 |
| 10 | ESN500_rad0.5_leak1.0_inp1.0_ridge0.0001 | 1.9530 | -2.1961 | 1 | 0 | 9.0 |

Use `REF_Ridge` at ~0.593 as reference (near-persistence).

## Lorenz63 Results (Top 10)

| Rank | Config | h=1 RMSE | Skill | VPT | FSDH | Time(s) |
|------|--------|----------|-------|-----|------|---------|
| 1 | ESN500_rad0.5_leak1.0_inp0.1_ridge1.0 | 0.0777 | 0.8168 | 2 | 2 | 7.1 |
| 2 | ESN500_rad0.5_leak1.0_inp1.0_ridge1.0 | 0.1609 | 0.6207 | 2 | 2 | 7.3 |
| 3 | ESN500_rad0.5_leak1.0_inp0.1_ridge0.0001 | 0.1635 | 0.6145 | 2 | 2 | 7.1 |
| 4 | ESN500_rad0.5_leak1.0_inp1.0_ridge0.0001 | 0.1747 | 0.5881 | 2 | 2 | 7.4 |
| 5 | ESN500_rad0.9_leak1.0_inp0.1_ridge1.0 | 0.1839 | 0.5664 | 2 | 2 | 7.3 |
| 6 | ESN500_rad1.5_leak1.0_inp1.0_ridge1.0 | 0.2093 | 0.5063 | 2 | 2 | 7.1 |
| 7 | ESN500_rad0.9_leak1.0_inp1.0_ridge1.0 | 0.2168 | 0.4888 | 2 | 2 | 7.2 |
| 8 | ESN500_rad0.9_leak1.0_inp1.0_ridge0.0001 | 0.2627 | 0.3806 | 2 | 2 | 7.1 |
| 9 | ESN500_rad0.9_leak1.0_inp0.1_ridge0.0001 | 0.2672 | 0.3699 | 2 | 2 | 7.2 |
| 10 | ESN500_rad1.5_leak1.0_inp1.0_ridge0.0001 | 0.3140 | 0.2596 | 2 | 2 | 7.2 |
