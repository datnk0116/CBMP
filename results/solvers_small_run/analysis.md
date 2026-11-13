# Analysis Report: solvers_small_run

## 1. Overview
- **Total records**: 100
- **Status**: 100% OPTIMAL
- **Duplicates**: 0
- **Validation**: 100% Valid (assignment consistency and objective match verified with relabel inversion)

## 2. Solver Performance Comparison
| Solver | Solved | Mean (s) | Median (s) | Min (s) | Max (s) | Total (s) |
|:---|---:|---:|---:|---:|---:|---:|
| cadical195 | 20 | 0.5941 | 0.3687 | 0.0001 | 2.1890 | 11.8812 |
| glucose4 | 20 | 0.8568 | 0.2047 | 0.0001 | 5.3054 | 17.1368 |
| lingeling | 20 | 2.1372 | 0.5305 | 0.0006 | 9.3126 | 42.7436 |
| maplechrono | 20 | 3.4924 | 3.5644 | 0.0002 | 19.2291 | 69.8475 |
| minisat22 | 20 | 0.5963 | 0.2283 | 0.0001 | 2.9836 | 11.9260 |

## 3. Objective Value Consistency
| Dataset | Seed | Objective | Solvers Checked | Status |
|:---|---:|---:|---:|:---|
| 14_bus | 0 | 3 | 5 | OK |
| 14_bus | 1 | 3 | 5 | OK |
| 14_bus | 2 | 3 | 5 | OK |
| 14_bus | 3 | 3 | 5 | OK |
| 14_bus | 4 | 3 | 5 | OK |
| 30_bus | 0 | 4 | 5 | OK |
| 30_bus | 1 | 4 | 5 | OK |
| 30_bus | 2 | 4 | 5 | OK |
| 30_bus | 3 | 4 | 5 | OK |
| 30_bus | 4 | 4 | 5 | OK |
| 494_bus_n40 | 0 | 5 | 5 | OK |
| 494_bus_n40 | 1 | 5 | 5 | OK |
| 494_bus_n40 | 2 | 5 | 5 | OK |
| 494_bus_n40 | 3 | 5 | 5 | OK |
| 494_bus_n40 | 4 | 5 | 5 | OK |
| 662_bus_n40 | 0 | 7 | 5 | OK |
| 662_bus_n40 | 1 | 7 | 5 | OK |
| 662_bus_n40 | 2 | 7 | 5 | OK |
| 662_bus_n40 | 3 | 7 | 5 | OK |
| 662_bus_n40 | 4 | 7 | 5 | OK |
