# Benchmark Datasets Manifest & Provenance

## 1. Overview & Verification

This directory contains benchmark instances used in the Cyclic Bandwidth Minimization Problem (CBMP) evaluation pipeline, stored in standard MatrixMarket coordinate format (`.mtx`), along with raw Harwell–Boeing `.mtx.gz` archives in `datasets/raw/`.

### Status Against Reference Literature (NILS / Ren et al. 2020, Table A.2)
- The repository workspace holds **42 `.mtx` files** on disk:
  - **28 Harwell–Boeing instances** matching the NILS Table A.2 list (`494_bus` and `662_bus` originally present, plus the 26 remaining instances fetched unmodified from NIST MatrixMarket via `scripts/fetch_harwell_boeing.py`).
  - **2 non-HB28 power bus graphs** (`14_bus`, `30_bus`) — IEEE test cases **not** part of the 28-instance HB list in NILS Table A.2.
  - **12 induced subgraphs** generated from `494_bus` and `662_bus` at sizes $N \in \{40, 50, 60, 80, 100, 150\}$ for scalability, ablation, and phase-transition analysis.
- **Scope Note**: The 85 standard/synthetic benchmark graphs from the 113-instance paper suite are not present in this repository and are outside the BTL scope; no claim is made of reproducing all 113 paper instances.
- **Integrity Guarantee**: Raw `.mtx` files downloaded from NIST MatrixMarket are preserved verbatim and **never modified** to force dimension or symmetry matches.

---

## 2. Harwell–Boeing 28 Verification Table (NILS Table A.2 vs Parsed MatrixMarket)

Parsed using `scipy.io.mmread` followed by self-loop removal ($u \ne v$), symmetrization, and undirected edge deduplication (raw `.mtx` files unmodified):

| Instance | MM Header | SymStruct | Expected $N$ | Expected $|E|$ | Parsed $N$ | Parsed $|E|$ | Match | Observed Notes |
|:---|:---:|:---:|---:|---:|---:|---:|:---:|:---|
| `jgl009` | `general` | no (symmetrized) | 9 | 50 | 9 | 32 | MISMATCH | `nnz=50` (8 diag, 42 off-diag directed -> 32 undirected edges) |
| `rgg010` | `general` | no (symmetrized) | 10 | 76 | 10 | 45 | MISMATCH | `nnz=76` (10 diag, 66 off-diag directed -> 45 undirected edges) |
| `jgl011` | `general` | no (symmetrized) | 11 | 76 | 11 | 49 | MISMATCH | `nnz=76` (10 diag, 66 off-diag directed -> 49 undirected edges) |
| `can_24` | `symmetric` | yes | 24 | 92 | 24 | 68 | MISMATCH | `diag=24`, `undir=68` (`68 + 24 = 92`) |
| `pores_1` | `general` | no (symmetrized) | 30 | 103 | 30 | 103 | OK | Asymmetric `.mtx`; requires symmetrization on load |
| `ibm32` | `general` | no (symmetrized) | 32 | 90 | 32 | 90 | OK | Asymmetric `.mtx`; requires symmetrization on load |
| `bcspwr01` | `symmetric` | yes | 39 | 46 | 39 | 46 | OK | Symmetric |
| `bcsstk01` | `symmetric` | yes | 48 | 176 | 48 | 176 | OK | Symmetric |
| `bcspwr02` | `symmetric` | yes | 49 | 59 | 49 | 59 | OK | Symmetric |
| `curtis54` | `general` | no (symmetrized) | 54 | 124 | 54 | 124 | OK | Asymmetric `.mtx`; requires symmetrization on load |
| `will57` | `general` | no (symmetrized) | 57 | 127 | 57 | 127 | OK | Asymmetric `.mtx`; requires symmetrization on load |
| `impcol_b` | `general` | no (symmetrized) | 59 | 281 | 59 | 240 | MISMATCH | `nnz=312` (17 diag, 254 off-diag directed -> 240 undirected edges) |
| `ash85` | `symmetric` | yes | 85 | 219 | 85 | 219 | OK | Symmetric |
| `nos4` | `symmetric` | yes | 100 | 247 | 100 | 247 | OK | Symmetric |
| `dwt_234` | `symmetric` | yes | 117 | 162 | 234 | 300 | MISMATCH | Full NIST `dwt__234` has $N=234, |E|=300$ (Table A.2 lists $117, 162$) |
| `bcspwr03` | `symmetric` | yes | 118 | 179 | 118 | 179 | OK | Symmetric |
| `bcsstk06` | `symmetric` | yes | 420 | 3720 | 420 | 3720 | OK | Symmetric |
| `bcsstk07` | `symmetric` | yes | 420 | 3720 | 420 | 3720 | OK | Symmetric |
| `impcol_d` | `general` | no (symmetrized) | 425 | 1267 | 425 | 1193 | MISMATCH | `nnz=1339` (28 diag, 1227 off-diag directed -> 1193 undirected edges) |
| `can_445` | `symmetric` | yes | 445 | 1682 | 445 | 1682 | OK | Symmetric |
| `494_bus` | `symmetric` | yes | 494 | 586 | 494 | 586 | OK | Symmetric (original repo file) |
| `dwt_503` | `symmetric` | yes | 503 | 2762 | 503 | 2762 | OK | Symmetric |
| `sherman4` | `general` | yes | 546 | 1341 | 1104 | 1341 | MISMATCH | $1104 \times 1104$ matrix with 546 non-isolated vertices and 1341 edges |
| `dwt_592` | `symmetric` | yes | 592 | 2256 | 592 | 2256 | OK | Symmetric |
| `662_bus` | `symmetric` | yes | 662 | 906 | 662 | 906 | OK | Symmetric (original repo file) |
| `nos6` | `symmetric` | yes | 675 | 1290 | 675 | 1290 | OK | Symmetric |
| `685_bus` | `symmetric` | yes | 685 | 1282 | 685 | 1282 | OK | Symmetric |
| `can_715` | `symmetric` | yes | 715 | 2975 | 715 | 2975 | OK | Symmetric |

Note on `graph_utils.load_graph_from_mtx`: by design (verified by `tests/test_graph_validation.py`), `load_graph_from_mtx` enforces structural symmetry on `.mtx` files and therefore raises `ValueError` on the 9 MatrixMarket `general` non-symmetric matrices (`jgl009`, `rgg010`, `jgl011`, `pores_1`, `ibm32`, `curtis54`, `will57`, `impcol_b`, `impcol_d`) unless symmetrized explicitly at caller level. We keep both the files and the strict validator unmodified.

---

## 3. Original 16 Core Benchmark & Induced Subgraph Instances

| Instance File | Graph Name | Vertices ($N$) | Edges ($|E|$) | Density ($2|E|/(N(N-1))$) | Provenance / Category |
|:---|:---|:---:|:---:|:---:|:---|
| `14_bus.mtx` | IEEE 14-bus | 14 | 20 | 0.2198 | Non-HB28 IEEE Power Bus |
| `30_bus.mtx` | IEEE 30-bus | 30 | 41 | 0.0943 | Non-HB28 IEEE Power Bus |
| `494_bus_n40.mtx` | 494-bus (N=40) | 40 | 47 | 0.0603 | Connected Induced Subgraph (Small) |
| `662_bus_n40.mtx` | 662-bus (N=40) | 40 | 50 | 0.0641 | Connected Induced Subgraph (Small) |
| `494_bus_n50.mtx` | 494-bus (N=50) | 50 | 60 | 0.0490 | Connected Induced Subgraph (Small) |
| `662_bus_n50.mtx` | 662-bus (N=50) | 50 | 60 | 0.0490 | Connected Induced Subgraph (Small) |
| `494_bus_n60.mtx` | 494-bus (N=60) | 60 | 72 | 0.0407 | Connected Induced Subgraph (Medium) |
| `662_bus_n60.mtx` | 662-bus (N=60) | 60 | 72 | 0.0407 | Connected Induced Subgraph (Medium) |
| `494_bus_n80.mtx` | 494-bus (N=80) | 80 | 96 | 0.0304 | Connected Induced Subgraph (Medium/Large) |
| `662_bus_n80.mtx` | 662-bus (N=80) | 80 | 102 | 0.0323 | Connected Induced Subgraph (Medium/Large) |
| `494_bus_n100.mtx` | 494-bus (N=100) | 100 | 120 | 0.0242 | Connected Induced Subgraph (Large) |
| `662_bus_n100.mtx` | 662-bus (N=100) | 100 | 129 | 0.0261 | Connected Induced Subgraph (Large) |
| `494_bus_n150.mtx` | 494-bus (N=150) | 150 | 185 | 0.0166 | Connected Induced Subgraph (Large) |
| `662_bus_n150.mtx` | 662-bus (N=150) | 150 | 196 | 0.0175 | Connected Induced Subgraph (Large) |
| `494_bus.mtx` | 494-bus (Full) | 494 | 586 | 0.0048 | Harwell–Boeing / Full Network (Large) |
| `662_bus.mtx` | 662-bus (Full) | 662 | 906 | 0.0041 | Harwell–Boeing / Full Network (Large) |

---

## 4. Validation

Verify all datasets referenced in `experiments/configs.yaml` and all 28 Harwell–Boeing dimensions:
```bash
./CBMP/bin/python scripts/check_datasets.py --config experiments/configs.yaml --dataset-dir datasets
./CBMP/bin/python scripts/fetch_harwell_boeing.py --verify-only
```
