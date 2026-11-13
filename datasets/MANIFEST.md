# Benchmark Datasets Manifest & Provenance

## 1. Overview & Verification

This directory contains benchmark instances used in the Cyclic Bandwidth Minimization Problem (CBMP) evaluation pipeline.
All datasets are stored in standard MatrixMarket coordinate format (`.mtx`), representing symmetric (`real symmetric`, lower-triangle storage; numeric values are ignored, so the graphs are treated as unweighted) adjacency matrices of undirected power network graphs. Every file also stores the `N` diagonal entries of the original sparse matrix; these self-loops are discarded by `graph_utils.load_graph_from_mtx`, so `|E|` below is the number of distinct off-diagonal undirected edges (MatrixMarket header entry count = `N + |E|`).

### Status Against Reference Literature
- In published literature (e.g. Martí et al., NILS benchmark suite), the "Harwell–Boeing" collection comprises **28 power system instances** (including bus network power systems and structural engineering matrices).
- This repository workspace contains **16 benchmark bus graph instances** on disk, validated by `scripts/check_datasets.py`:
  - **4 full power bus graphs** from the standard Harwell–Boeing collection (`14_bus`, `30_bus`, `494_bus`, `662_bus`).
  - **12 induced subgraphs** systematically generated from `494_bus` and `662_bus` at sizes $N \in \{40, 50, 60, 80, 100, 150\}$ for scalability, ablation, and phase-transition analysis.
- **Integrity Guarantee**: We strictly document actual dataset provenance and **do not fabricate** synthetic or missing instances. All experiments in `experiments/configs.yaml` reference solely the 16 verified instances.

---

## 2. Manifest of Instances

| Instance File | Graph Name | Vertices ($N$) | Edges ($|E|$) | Density ($2|E|/(N(N-1))$) | Provenance / Category |
|:---|:---|:---:|:---:|:---:|:---|
| `14_bus.mtx` | IEEE 14-bus | 14 | 20 | 0.2198 | Harwell–Boeing / Full Network |
| `30_bus.mtx` | IEEE 30-bus | 30 | 39 | 0.0897 | Harwell–Boeing / Full Network |
| `494_bus_n40.mtx` | 494-bus (N=40) | 40 | 52 | 0.0667 | Connected Induced Subgraph (Small) |
| `662_bus_n40.mtx` | 662-bus (N=40) | 40 | 50 | 0.0641 | Connected Induced Subgraph (Small) |
| `494_bus_n50.mtx` | 494-bus (N=50) | 50 | 65 | 0.0531 | Connected Induced Subgraph (Small) |
| `662_bus_n50.mtx` | 662-bus (N=50) | 50 | 64 | 0.0522 | Connected Induced Subgraph (Small) |
| `494_bus_n60.mtx` | 494-bus (N=60) | 60 | 76 | 0.0429 | Connected Induced Subgraph (Medium) |
| `662_bus_n60.mtx` | 662-bus (N=60) | 60 | 77 | 0.0435 | Connected Induced Subgraph (Medium) |
| `494_bus_n80.mtx` | 494-bus (N=80) | 80 | 97 | 0.0307 | Connected Induced Subgraph (Medium/Large) |
| `662_bus_n80.mtx` | 662-bus (N=80) | 80 | 105 | 0.0332 | Connected Induced Subgraph (Medium/Large) |
| `494_bus_n100.mtx` | 494-bus (N=100) | 100 | 118 | 0.0238 | Connected Induced Subgraph (Large) |
| `662_bus_n100.mtx` | 662-bus (N=100) | 100 | 130 | 0.0263 | Connected Induced Subgraph (Large) |
| `494_bus_n150.mtx` | 494-bus (N=150) | 150 | 175 | 0.0157 | Connected Induced Subgraph (Large) |
| `662_bus_n150.mtx` | 662-bus (N=150) | 150 | 190 | 0.0170 | Connected Induced Subgraph (Large) |
| `494_bus.mtx` | 494-bus (Full) | 494 | 586 | 0.0048 | Harwell–Boeing / Full Network (Large) |
| `662_bus.mtx` | 662-bus (Full) | 662 | 906 | 0.0041 | Harwell–Boeing / Full Network (Large) |

---

## 3. Validation

All 16 instances are checked automatically via:
```bash
./CBMP/bin/python scripts/check_datasets.py --config experiments/configs.yaml --dataset-dir datasets
```
Which confirms 100% presence and integrity without missing dependencies.

The `|E|` and density columns above are asserted against the actual files by `tests/test_fix_audit4_regressions.py::test_manifest_edge_counts_match_files`.
