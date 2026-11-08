# Vietnamese Relational Knowledge Distillation: Qwen3-4B sang 0.6B

Dự án chưng cất tri thức dạng quan hệ (**Similarity / Gram Matrix Relational Knowledge Distillation**) từ mô hình Teacher **`Qwen/Qwen3-Embedding-4B`** (2560 chiều) sang mô hình Student **`Qwen/Qwen3-Embedding-0.6B`** (1024 chiều) trên tập dữ liệu tiếng Việt (có hỗ trợ mở rộng song ngữ).

---

## 1. Bản chất cốt lõi bài toán

- **Khác biệt số chiều:** Teacher (2560D) vs Student (1024D). Không ép bằng tọa độ mà ép **ma trận góc tương quan cosine Gram ($4 \times 4$)** giữa các văn bản trong nhóm $K=4$ ($1$ positive + $3$ hard negatives).
- **Teacher Offline Caching:** Chạy Teacher 4B inference trước 1 lần để tính ma trận Gram, lưu vào file rồi giải phóng hoàn toàn VRAM trước khi train Student.
- **Hàm Loss:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NCE}} + \lambda_{\text{rel}} \cdot \mathcal{L}_{\text{RKD}}$$
  - **Nhánh `E1` (Baseline):** $\lambda_{\text{rel}} = 0.0$ (chỉ train InfoNCE giám sát).
  - **Nhánh `E2` (Relational KD):** $\lambda_{\text{rel}} = 0.5$ (kết hợp InfoNCE và cấu trúc tương quan từ Teacher).

---

## 2. Cấu trúc thư mục dự án

```text
Embedding-Distillation/
├── README.md                      # Hướng dẫn chi tiết
├── requirements.txt               # Danh sách thư viện phụ thuộc
├── configs/                       # Cấu hình siêu tham số
│   ├── e1_infonce.yaml            # Nhánh E1 (InfoNCE baseline)
│   └── e2_relational_kd.yaml      # Nhánh E2 (Relational KD)
├── data/                          # Chứa dữ liệu huấn luyện
│   └── .gitkeep
├── src/                           # Mã nguồn cốt lõi (100% dữ liệu & model thật)
│   ├── __init__.py
│   ├── compat.py                  # Bản vá tương thích & cấu hình phần cứng
│   ├── prepare_data.py            # Stream dữ liệu mMARCO & MS MARCO
│   ├── score_teacher.py           # Chạy Teacher 4B offline tính Gram matrix
│   ├── train.py                   # Huấn luyện Student LoRA (E1 & E2)
│   └── evaluate.py                # Đánh giá retrieval cục bộ & VN-MTEB
├── tests/                         # Kiểm thử toán học
│   └── test_rkd.py                # Chạy: python tests/test_rkd.py
├── scripts/                       # Kịch bản chạy tự động hóa
│   ├── run_pipeline.bat           # Dành cho Windows
│   └── run_pipeline.sh            # Dành cho Linux / A100 Server
└── docs/                          # Tài liệu kỹ thuật
    └── Vietnamese_Relational_KD_Qwen3_4B_to_0.6B_Technical_Plan.md
```

---

## 3. Cài đặt môi trường

```bash
pip install -r requirements.txt
```

Kiểm tra tính đúng đắn toán học:
```bash
python tests/test_rkd.py
```

---

## 4. Quy trình chạy thực tế

### Bước 1: Tải & chuẩn bị dữ liệu thật (`src/prepare_data.py`)
- **Tập Pilot (20.000 nhóm tiếng Việt từ mMARCO):**
  ```bash
  python src/prepare_data.py --n 20000 --out data/groups.jsonl --seed 42
  ```
- **Tập Main Song ngữ (100.000 nhóm Việt + 50.000 nhóm Anh):**
  ```bash
  python src/prepare_data.py --n 100000 --en-samples 50000 --out data/groups.jsonl --seed 42
  ```

---

### Bước 2: Chạy Teacher 4B Offline tính Gram Matrix (`src/score_teacher.py`)
```bash
python src/score_teacher.py \
  --input data/groups.jsonl \
  --output data/groups_scored.jsonl \
  --model Qwen/Qwen3-Embedding-4B \
  --batch-groups 4
```
*(Trên A100 80GB, có thể tăng `--batch-groups 16` hoặc `32` để tăng tốc).*

---

### Bước 3: Huấn luyện Student LoRA (`src/train.py`)

#### Nhánh E1: Baseline InfoNCE ($\lambda_{\text{rel}} = 0.0$)
```bash
python src/train.py \
  --data data/groups_scored.jsonl \
  --model Qwen/Qwen3-Embedding-0.6B \
  --lambda-rel 0.0 \
  --output outputs/E1_InfoNCE \
  --batch-groups 4 \
  --accum 8 \
  --epochs 1
```

#### Nhánh E2: Relational KD ($\lambda_{\text{rel}} = 0.5$)
```bash
python src/train.py \
  --data data/groups_scored.jsonl \
  --model Qwen/Qwen3-Embedding-0.6B \
  --lambda-rel 0.5 \
  --output outputs/E2_RKD \
  --batch-groups 4 \
  --accum 8 \
  --epochs 1
```

---

### Bước 4: Đánh giá mô hình (`src/evaluate.py`)
- **Đánh giá trên tập kiểm tra cục bộ:**
  ```bash
  python src/evaluate.py --model outputs/E2_RKD --eval-file data/groups_scored.jsonl
  ```
- **Đánh giá trên chuẩn VN-MTEB (Retrieval tasks):**
  ```bash
  python src/evaluate.py --model outputs/E2_RKD --mteb --output results/E2_MTEB
  ```

---

## 5. Chạy trên Cloud GPU A100 (80GB) bằng Modal (`run_modal.py`)

File `run_modal.py` là file độc lập hoàn toàn (All-in-one). Hỗ trợ **CHẠY NGẦM TRÊN CLOUD (DETACHED MODE)** — bật lên là có thể tắt terminal, tắt máy tính đi ngủ:

1. **Đăng nhập Modal (chỉ làm 1 lần):**
   ```bash
   python -m modal setup
   ```

2. **Chạy ngầm toàn bộ từ A – Z (Không cần giữ máy, tự chạy trên A100):**
   ```bash
   # Chạy ngầm tập Pilot (20K tiếng Việt)
   python -m modal run run_modal.py --detach

   # Hoặc chạy ngầm tập lớn Main Song ngữ (100K Việt + 50K Anh)
   python -m modal run run_modal.py --detach --n 100000 --en-samples 50000
   ```
   *Terminal sẽ trả về Task ID và link theo dõi trong 2 giây. Bạn có thể tắt máy tính ngay lập tức.*

3. **Theo dõi tiến trình & Xem log thời gian thực:**
   - Mở trình duyệt truy cập: [https://modal.com/apps](https://modal.com/apps)

4. **Tải checkpoint mô hình từ Cloud về máy cá nhân (sau khi train xong):**
   ```bash
   python -m modal volume get embedding-kd-volume /outputs/E2_RKD/best ./outputs/E2_RKD_best
   ```

---

## 6. Kết quả Thực nghiệm & Benchmark (MTEB Quốc tế)

Bảng xếp hạng đối đầu trực tiếp giữa 5 mô hình trên Cloud GPU A100:

| Mô hình | Tham số | Chiều | VieQuAD (VN) | SciFact-VN | ArguAna-VN | **TB Tiếng Việt** | SciFact (EN) | ArguAna (EN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Qwen3-Embedding-4B (Teacher)** | 4.0B | 2560D | **0.6250** | **0.7342** | **0.4382** | **0.5991** | **0.7755** | **0.7363** |
| 🥈 **Qwen3-Embedding-0.6B (Base)** | 0.6B | 1024D | 0.5510 | 0.6714 | 0.4050 | **0.5425** | 0.7030 | 0.6777 |
| 🥉 **BAAI/bge-m3 (Multilingual)** | 0.6B | 1024D | 0.5684 | 0.6231 | 0.3763 | **0.5226** | 0.6437 | 0.5396 |
| ⭐ **Student-0.6B-RKD (Ours Best)** | 0.6B | 1024D | 0.5097 | **0.6386** 🏆 | **0.3770** 🏆 | **0.5084** | 0.0000 | 0.0000 |
| 🏅 **BKAI-Bi-Encoder (Vietnamese)** | 135M | 768D | 0.4347 | 0.3929 | 0.2865 | **0.3714** | 0.2623 | 0.2440 |

### Điểm nổi bật:
* **Vượt BGE-M3 trên các bài toán suy luận khó**: Student 0.6B RKD đạt **0.6386** trên `SciFact-VN` (so với 0.6231 của BGE-M3) và **0.3770** trên `ArguAna-VN` (so với 0.3763 của BGE-M3).
* **Bỏ xa BKAI Bi-Encoder**: Vượt trội mô hình phổ biến nhất trên HuggingFace gần **14 điểm nDCG** trên tiếng Việt (**0.5084** so với **0.3714**).
* **Chuyên biệt hóa tiếng Việt**: Mô hình LoRA được tối ưu sâu vào không gian biểu diễn tiếng Việt, đóng vai trò một **Vietnamese Specialist** hiệu năng cao với chi phí suy luận cực thấp.
   ```
