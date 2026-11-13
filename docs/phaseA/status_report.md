# CBMP BTL số 2 - Thông tin thực nghiệm

## 1. Yêu cầu & Trạng thái
- **Đề tài**: Cyclic Bandwidth Minimization Problem (CBMP) - SAT Encoding.
- **Yêu cầu chính**: Pairwise encoding, Order encoding, so sánh baselines (CPLEX MIP, CPLEX CP, Gurobi, CP-SAT), báo cáo khoa học.
- **Trạng thái thực nghiệm**:
    - **Dataset**: Đã xác nhận 16 instance thực tế (không giả lập, không bịa đặt).
    - **Code lõi**: Đã audit ổn định, vượt qua 1047 tests (1 skipped).
    - **License Solver**:
        - Gurobi: Cần kiểm tra bản quyền (Hiện chưa cài đặt `gurobipy`).
        - CPLEX: Bản Community 22.2.0.0 (giới hạn ~1000 biến/ràng buộc).
    - **Cấu trúc báo cáo**: Đang sửa theo template `cas-sc` (cần cập nhật tên tác giả).

## 2. Các hành động cần thực hiện tiếp theo
1. **License & Solver**: 
    - Kiểm tra `gurobipy` (đã xác định thiếu).
    - Cài đặt/Sửa `scripts/estimate_model_size.py` để dựng matrix kích thước mô hình (dùng logic đã có trong các file `baselines/*_cbmp.py`).
2. **Thực nghiệm**:
    - Chạy thử nghiệm tầng (1): Nhóm nhỏ (`solvers_small`).
    - Chạy thử nghiệm tầng (2): Nhóm trung bình (subgraph n80–n150).
    - Chạy thử nghiệm tầng (3): Các instance lớn (494_bus, 662_bus).
3. **Báo cáo**:
    - Cập nhật thông tin tác giả trong `report.tex`.
    - Vẽ hình bằng `cnsplots` (cần cài đặt và test API).

---
*Ghi chú*: Mọi kết quả thực nghiệm phải dựa trên bản ghi JSONL thực tế. Không sử dụng các kết quả giả lập hay kết quả cũ nếu code đã thay đổi.
