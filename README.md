# Cyclic Bandwidth Minimization Problem (CBMP) Solver

A research and evaluation repository for solving the **Cyclic Bandwidth Minimization Problem (CBMP)** on undirected graphs using Boolean Satisfiability (SAT), Constraint Programming (OR-Tools CP-SAT), and Mixed Integer Linear Programming (IBM ILOG CPLEX via DOcplex).

---

## 1. Cài đặt (Installation)

### Yêu cầu hệ thống
- Python 3.10+ (đã kiểm thử trên Python 3.14 macOS arm64).
- Bộ giải SAT: `pysat` (tích hợp CaDiCaL 195, Glucose 4, MapleChrono, MiniSat 2.2, Lingeling).
- CP-SAT: `ortools`.
- MIP: `docplex` và `cplex` (Community Edition giới hạn <= 1000 biến & ràng buộc hoặc bản thương mại/học thuật đầy đủ).

### Khởi tạo môi trường ảo và cài đặt thư viện
```bash
# Tạo môi trường ảo
python3 -m venv venv
source venv/bin/activate

# Cài đặt các gói theo requirements.txt (đã ghim phiên bản chuẩn)
pip install -r requirements.txt
```

Các phiên bản đã ghim tương thích với `results/env.json`:
- `python-sat==1.9.dev21`
- `scipy==1.18.1`
- `numpy==2.5.4`
- `ortools==9.15.6755`
- `docplex==2.32.264`
- `cplex==22.2.0.0`
- `pytest==9.2.0`
- `pyyaml==6.0.3`
- `pandas>=2.0.0`
- `matplotlib>=3.8.0`
- `networkx>=3.0`

---

## 2. Cấu trúc thư mục (Directory Structure)

```
CBMP/
├── cbmp.py                 # Core SAT encoding & giải thuật lặp thu hẹp Cyclic Bandwidth
├── main.py                 # CLI giải 1 instance với timeout & multiprocessing isolation
├── sat_encodings.py        # Các mệnh đề CNF (bậc thang, injectivity, phá đối xứng, cyclic edge)
├── baselines/
│   ├── cpsat_cbmp.py       # Baseline OR-Tools CP-SAT
│   └── cplex_cbmp.py       # Baseline IBM CPLEX MIP
├── experiments/
│   ├── configs.yaml        # Cấu hình các nhóm thực nghiệm (per_dataset, solvers, methods, ablation)
│   └── run_experiments.py # Trình điều khiển thực nghiệm (deduplication, resuming, runner)
├── analysis/
│   ├── make_tables.py      # Sinh bảng LaTeX và CSV (ablation, solvers, methods)
│   └── make_plots.py       # Vẽ biểu đồ cactus, scatter, ablation
├── datasets/               # Thư mục chứa ma trận đồ thị MatrixMarket (.mtx)
│   ├── 14_bus.mtx
│   ├── 30_bus.mtx
│   ├── 494_bus.mtx
│   └── 662_bus.mtx
├── scripts/
│   └── check_datasets.py   # Kiểm tra và cảnh báo nếu dataset trong config bị thiếu
├── tests/                  # Bộ test tự động (brute force, baselines, graph atlas, correctness)
│   ├── brute_force.py
│   ├── test_baselines.py
│   ├── test_correctness.py
│   ├── test_experiments.py
│   └── test_graph_atlas.py
├── results/                # Kết quả chạy thực nghiệm (runs.jsonl, tables/, figures/, env.json)
├── legacy/                 # Các mã nguồn cũ không còn được import
│   └── bmp.py              # Phiên bản cũ của bộ giải (đã thay thế bởi cbmp.py)
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## 3. Cách chạy `main.py` cho 1 instance

`main.py` nhận tên dataset (trong thư mục `datasets/`) hoặc đường dẫn file `.mtx`:

```bash
# Chạy với cấu hình tốt nhất (Rotation + Reflection symmetry breaking, Heuristic UB, CaDiCaL 195)
python main.py 30_bus --symmetry-break --heuristic-ub --timeout 60

# Chạy không dùng phá đối xứng và không dùng heuristic (trivial bound N//2+1)
python main.py 30_bus --timeout 60

# Chọn solver SAT khác (glucose4, maplechrono, minisat22, lingeling)
python main.py 30_bus --symmetry-break --solver glucose4 --timeout 60

# Lưu kết quả ra file JSON line
python main.py 30_bus --symmetry-break --heuristic-ub --timeout 60 --json-out results/single_run.jsonl
```

### Các cờ chính của `main.py`:
- `dataset`: Tên file (ví dụ `30_bus` hoặc `datasets/30_bus.mtx`).
- `--timeout`: Thời gian tối đa (giây), mặc định: 600.
- `--symmetry-break`: Bật cả phá đối xứng quay (`--sb-rotation`) và đối xứng gương (`--sb-reflection`).
- `--heuristic-ub`: Sử dụng nhãn xuất phát từ thuật toán Cuthill-McKee để khởi tạo cận trên.
- `--solver`: Trình giải SAT (mặc định: `cadical195`).
- `--json-out`: Đường dẫn file để ghi kết quả dưới dạng JSON line.

---

## 4. Kiểm tra dữ liệu (Dataset Validation)

Trước khi chạy thực nghiệm, bạn có thể kiểm tra xem tất cả các dataset được tham chiếu trong file cấu hình đã tồn tại hay chưa:

```bash
python scripts/check_datasets.py --config experiments/configs.yaml --dataset-dir datasets
```

Nếu thiếu file (ví dụ chưa có `14_bus.mtx`), script sẽ báo lỗi rõ ràng và liệt kê chính xác các file cần bổ sung.

---

## 5. Chạy thực nghiệm (Running Experiments)

File cấu hình tại `experiments/configs.yaml` định nghĩa 4 nhóm thực nghiệm. Runner `experiments/run_experiments.py` tự động khử trùng lặp các tác vụ dùng chung cấu hình.

### Xem trước kế hoạch chạy (Dry Run)
```bash
python experiments/run_experiments.py --dry-run
```

### Chạy từng nhóm thực nghiệm cụ thể

```bash
# 1. Nhóm per_dataset (Cấu hình SAT tốt nhất trên toàn bộ dataset)
python experiments/run_experiments.py --group per_dataset --timeout 600

# 2. Nhóm solvers (So sánh các SAT solver khác nhau)
python experiments/run_experiments.py --group solvers --timeout 600

# 3. Nhóm methods (So sánh SAT vs OR-Tools CP-SAT vs IBM CPLEX)
python experiments/run_experiments.py --group methods --timeout 600

# 4. Nhóm ablation (Đánh giá đóng góp của từng thành phần: quay, phản xạ, heuristic)
python experiments/run_experiments.py --group ablation --timeout 600

# Chạy tất cả các nhóm kèm cờ resume (bỏ qua những lần chạy đã hoàn thành)
python experiments/run_experiments.py --group all --resume

# Ghi thông tin commit, timeout, và config vào metadata
python experiments/run_experiments.py --group all --record-metadata
```

> [!NOTE]
> **Ghi chú về kết quả bảng biểu hiện tại (`results/tables/`)**:
> Các bảng tổng hợp hiện tại trong `results/tables/` được chạy với `--timeout 10` (có thể kiểm chứng từ giá trị PAR2 của các lượt timeout là 2 × 10 = 20.0s).
> Sử dụng cờ `--record-metadata` khi chạy runner để tự động lưu git commit hash, timeout, và đường dẫn cấu hình vào `results/env.json` cũng như đầu file `results/runs.jsonl`.

---

## 6. Sinh bảng biểu và đồ thị (Analysis & Visualization)

```bash
# Sinh các bảng tổng hợp LaTeX (.tex) và CSV (.csv) vào results/tables/
python analysis/make_tables.py results/runs.jsonl

# Sinh các biểu đồ cactus, scatter, ablation vào results/figures/
python analysis/make_plots.py results/runs.jsonl
```

- Bảng `table_ablation`: Đánh giá tác động của phá đối xứng quay, đối xứng gương và heuristic khởi tạo.
- Bảng `table_solvers`: So sánh thời gian và tỷ lệ giải được của các solver SAT.
- Bảng `table_methods`: So sánh SAT với CP-SAT và CPLEX. Trường hợp CPLEX chạm ngưỡng Community Edition (<= 1000 biến/ràng buộc) được đánh dấu riêng bằng mã **SL** (Size Limit), tách biệt khỏi PAR2 và được chú thích dưới bảng.

---

## 7. Chạy kiểm thử tự động (Running Tests)

Toàn bộ các thuật toán và mô hình được kiểm thử tự động bằng `pytest`:

```bash
# Chạy bộ test tiêu chuẩn (loại bỏ các bài test nặng, hoàn thành trong ~3 giây)
pytest -q -m "not slow"

# Chạy đầy đủ cả các bài test exhaustive graph atlas (1..7 đỉnh, ~60 giây)
pytest -q

# Chạy riêng từng file test:
pytest tests/test_correctness.py -v   # Tính đúng đắn của SAT encoding & phá đối xứng
pytest tests/test_baselines.py -v     # Kiểm thử CP-SAT & CPLEX baselines
pytest tests/test_experiments.py -v   # Kiểm thử lập kế hoạch, tổng hợp số liệu
pytest tests/test_graph_atlas.py -v   # Đối chiếu toàn diện 1.252 đồ thị với Brute Force
```

---

## 8. Ghi chú mã nguồn cũ (Legacy)

- `legacy/bmp.py`: Là phiên bản triển khai CBMP đời đầu, không còn module nào trong dự án phụ thuộc hoặc import file này. Đã được chuyển vào thư mục `legacy/` để bảo tồn lịch sử phát triển.

