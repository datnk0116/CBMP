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

File cấu hình tại `experiments/configs.yaml` định nghĩa 6 nhóm thực nghiệm. Runner `experiments/run_experiments.py` tự động khử trùng lặp các tác vụ dùng chung cấu hình. Mặc định toàn bộ kết quả được ghi vào thư mục `results_v2/` (đã được cấu hình trong `.gitignore`). Bạn có thể tùy chỉnh thư mục xuất qua cờ `--output-dir`.

### Xem trước kế hoạch chạy (Dry Run)
```bash
python experiments/run_experiments.py --dry-run
```

### Chạy từng nhóm thực nghiệm cụ thể

```bash
# 1. Nhóm per_dataset (Cấu hình SAT tốt nhất trên toàn bộ dataset)
python experiments/run_experiments.py --group per_dataset --timeout 600

# 2. Nhóm solvers_small (So sánh các SAT solver trên đồ thị nhỏ: N <= 40, 5 reps)
python experiments/run_experiments.py --group solvers_small --timeout 600

# 3. Nhóm solvers_medium (So sánh các SAT solver trên đồ thị trung bình: N = 60, 80, 1 rep)
python experiments/run_experiments.py --group solvers_medium --timeout 600

# 4. Nhóm methods (So sánh SAT vs OR-Tools CP-SAT vs IBM CPLEX trên đồ thị nhỏ & trung bình)
python experiments/run_experiments.py --group methods --timeout 600

# 5. Nhóm ablation_small (Đánh giá đóng góp quay, phản xạ, heuristic trên đồ thị nhỏ, 5 reps)
python experiments/run_experiments.py --group ablation_small --timeout 600

# 6. Nhóm ablation_medium (Đánh giá ablation trên đồ thị biên trung bình 662_bus_n60, 1 rep)
python experiments/run_experiments.py --group ablation_medium --timeout 600

# 7. Nhóm ablation_family (Đánh giá ablation trên các họ hoán vị nhãn ngẫu nhiên: n40 với 10 seeds, n50 với 5 seeds, 1 rep)
python experiments/run_experiments.py --group ablation_family --timeout 600

# Chỉ định thư mục lưu kết quả tùy chỉnh qua --output-dir
python experiments/run_experiments.py --group solvers_small --output-dir results_v2

# Chạy kèm cờ resume (mặc định chỉ bỏ qua các record đã chạy trong runs.jsonl)
python experiments/run_experiments.py --group all --resume

# Chạy lại các bài toán bị thất bại (TIMEOUT hoặc ERROR) từ runs.jsonl
python experiments/run_experiments.py --group all --resume --retry-failed

# Ép buộc resume khi git commit hash hiện tại khác với commit trong runs.jsonl
python experiments/run_experiments.py --group all --resume --force-resume

# Ghi thông tin commit, timeout, và config vào metadata
python experiments/run_experiments.py --group all --record-metadata
```

> [!TIP]
> **Cơ chế `--resume`, `--force-resume`, và `--retry-failed`:**
> - Mỗi bản ghi trong `runs.jsonl` được tự động gắn git commit hash và cờ repo dirty (`git_commit`, `git_dirty`).
> - `--resume`: Kiểm tra commit hash của các kết quả cũ. Nếu commit hash khác với commit hiện tại của repo, runner sẽ từ chối chạy để tránh trộn lẫn kết quả của các phiên bản mã nguồn khác nhau.
> - `--force-resume`: Cho phép bỏ qua kiểm tra commit hash và tiếp tục chạy resume.
> - `--retry-failed`: Mặc định `--resume` bỏ qua tất cả kết quả đã ghi. Khi thêm `--retry-failed`, runner sẽ chỉ bỏ qua các record `OPTIMAL` (hoặc `SIZE_LIMIT`), và tự động chạy lại các cấu hình trước đó bị `TIMEOUT` hoặc `ERROR`.

> [!NOTE]
> Mặc định runner ghi dữ liệu vào `results_v2/` và không can thiệp vào `results/`.
> Sử dụng cờ `--record-metadata` khi chạy runner để tự động lưu git commit hash, timeout, và đường dẫn cấu hình vào `results_v2/env.json` cũng như đầu file `results_v2/runs.jsonl`.

---

## 6. Sinh bảng biểu và đồ thị (Analysis & Visualization)

Các script phân tích mặc định đọc từ `results_v2/runs.jsonl` và xuất ra `results_v2/tables/` cũng như `results_v2/figures/`. Có thể tùy chỉnh thư mục đầu ra thông qua cờ `--output-dir`:

```bash
# Sinh các bảng tổng hợp LaTeX (.tex) và CSV (.csv) vào results_v2/tables/
python analysis/make_tables.py results_v2/runs.jsonl
# Hoặc chỉ định rõ thư mục xuất:
python analysis/make_tables.py results_v2/runs.jsonl --output-dir results_v2/tables

# Sinh các biểu đồ cactus, scatter, ablation vào results_v2/figures/
python analysis/make_plots.py results_v2/runs.jsonl
# Hoặc chỉ định rõ thư mục xuất:
python analysis/make_plots.py results_v2/runs.jsonl --output-dir results_v2/figures
```

- Bảng `table_ablation_small` & `table_ablation_medium`: Đánh giá tác động của phá đối xứng quay, đối xứng gương và heuristic khởi tạo (tách riêng đồ thị nhỏ và trung bình).
- Bảng `table_ablation_family`: Đánh giá phân phối tỷ lệ thời gian giữa các biến thể trên họ các đồ thị hoán vị nhãn ngẫu nhiên (seeds), cung cấp median tỷ lệ thời gian, IQR (25-75%), min-max, và tỷ lệ số cặp nhanh hơn.
- Bảng `table_ablation_family_per_graph`: Chi tiết các so sánh cặp trên từng đồ thị con họ bus ($N=40, 50$).
- Bảng `table_ablation_phases`: Phân tích thời gian giải theo từng giai đoạn (vòng SAT cuối cùng `last_iter_time` vs các vòng trước `prev_iters_time`), phân tách theo $N=40$ và $N=50$.
- Bảng `table_solvers_small` & `table_solvers_medium`: So sánh thời gian và tỷ lệ giải được của 5 solver SAT trên đồ thị nhỏ và trung bình.
  - *Ghi chú cho `table_solvers_small`*: Mỗi dataset được chạy trên 5 hoán vị nhãn ngẫu nhiên (`relabel_seeds` 0..4). Thời gian giải được gộp theo trung vị (**median**); điểm PAR2 được tính theo trung bình cộng (**mean**) trên 5 seed.
- Bảng `table_methods`: So sánh SAT với CP-SAT và CPLEX trên 7 instance (nhỏ + trung bình). Trường hợp CPLEX chạm ngưỡng Community Edition (<= 1000 biến/ràng buộc) được đánh dấu riêng bằng mã **SL** (Size Limit), tách biệt khỏi PAR2 và được chú thích dưới bảng.
  - *Quy tắc Speedup & Mean time*: Tốc độ tăng trung bình nhân (Speedup) chỉ tính trên các instance mà baseline $\ge 0.1\text{s}$ (kèm hiển thị số instance đủ điều kiện, ví dụ `[5/7]`). Nếu solver/method chưa giải được hết các instance này thì ghi `-`. Nếu một method chưa giải hết toàn bộ instance trong bảng thì dòng "mean time" ghi `-`.
- Bảng `table_methods_family`: So sánh hiệu năng giữa CP-SAT và SAT best trên 6 đồ thị có hoán vị nhãn ngẫu nhiên (seeds 0..4 và 0..2): median tỷ lệ thời gian `cpsat/sat`, min, max, và số cặp CP-SAT nhanh hơn / tổng số cặp.
- Bảng `table_methods_large`: So sánh CP-SAT và SAT best trên các đồ thị lớn ($N \ge 80$, nhãn gốc) gồm status, best_ub, best_lb (từ CP-SAT hoặc `lb_degree = ceil(Delta/2)`) và thời gian giải.
- Bảng `table_per_dataset`: Tổng hợp kết quả SAT best (CaDiCaL 195) trên toàn bộ 16 dataset MatrixMarket, bổ sung cận dưới $LB = \max(LB_{\text{degree}}, LB_{\text{solver}})$ và khoảng cách $Gap = UB - LB$ cho các dòng TIMEOUT.
- Bảng `cross_validation.csv`: Đối chiếu nghiệm tối ưu `best_ub` giữa tất cả solver/method cùng đạt trạng thái OPTIMAL trên cùng dataset/seed (cột `relabel_seed` kiểu Int64, để trống cho nhãn gốc). Báo lỗi và dừng nếu phát hiện mâu thuẫn hoặc nghiệm incumbent không hợp lệ (< opt).

---

## 7. Hoàn thiện kết quả (Finalize)

Khi toàn bộ thực nghiệm sạch trong `results_v2/` đã hoàn thành và được kiểm chứng hợp lệ:
1. Lưu trữ kết quả pilot cũ (chạy trên 4 đồ thị: `14_bus`, `30_bus`, `494_bus`, `662_bus`, timeout 60s, không ghi `git_commit`):
   - Để tránh lồng thư mục (do `results/` cũ đang chứa thư mục con `results/pilot/` bên trong):
     ```bash
     # Nếu results/pilot tồn tại, tách riêng thư mục pilot trước:
     git mv results/pilot results_pilot

     # Sau đó di chuyển phần còn lại (bảng biểu, đồ thị cũ):
     git mv results results_pilot_tables
     ```
     *(Ghi chú: Nếu di chuyển trực tiếp cả thư mục thì dùng `git mv results results_pilot_old`, chứa results/pilot/ lồng bên trong).*
2. Sao chép kết quả chính thức từ `results_v2/` sang `results/` và commit:
   ```bash
   cp -r results_v2 results
   git add results_pilot results_pilot_tables results
   git commit -m "chore: archive pilot results and commit finalized experiment results"
   ```

---

## 8. Chạy kiểm thử tự động (Running Tests)

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

## 9. Ghi chú mã nguồn cũ (Legacy)

- `legacy/bmp.py`: Là phiên bản triển khai CBMP đời đầu, không còn module nào trong dự án phụ thuộc hoặc import file này. Đã được chuyển vào thư mục `legacy/` để bảo tồn lịch sử phát triển.

