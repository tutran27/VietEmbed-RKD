# Vietnamese Relational Knowledge Distillation: Qwen3-4B sang 0.6B

Dự án chưng cất tri thức dạng quan hệ (**Similarity / Gram Matrix Relational Knowledge Distillation**) từ mô hình Teacher **`Qwen/Qwen3-Embedding-4B`** (2560 chiều) sang mô hình Student **`Qwen/Qwen3-Embedding-0.6B`** (1024 chiều) trên tập dữ liệu tiếng Việt.

---

## 1. Giới thiệu bài toán & Công thức cốt lõi

### Vì sao dùng Relational KD thay vì ép tọa độ MSE?
- **Khác biệt số chiều vector:** Teacher (2560D) và Student (1024D). Không thể ép trực tiếp tọa độ (MSE feature) mà không dùng thêm ma trận chiếu (projection layer). Thay vào đó, Relational KD bảo toàn **cấu trúc hình học tương quan cosine** giữa các văn bản.
- **Ma trận Gram góc tương quan:** Với mỗi nhóm gồm $K = 4$ văn bản (1 positive + 3 negatives), ma trận embedding sau khi chuẩn hóa $L_2$ là $Z \in \mathbb{R}^{K \times D}$. Ma trận Gram tương quan cosine $G \in \mathbb{R}^{K \times K}$ được tính bởi:

$$
G = Z Z^\top \in \mathbb{R}^{K \times K}, \quad \text{với } G_{jk} = \cos(z_j, z_k) = z_j \cdot z_k
$$

- **Teacher Offline Caching:** Chạy Teacher inference 1 lần để sinh ma trận Gram $G_T$, lưu vào file rồi giải phóng hoàn toàn VRAM trước khi huấn luyện Student.

---

### Hàm Loss tổng quát

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NCE}} + \lambda_{\text{rel}} \cdot \mathcal{L}_{\text{RKD}}
$$

### Chi tiết các thành phần Loss

#### 1. InfoNCE Loss ($\mathcal{L}_{\text{NCE}}$)
Phân loại tài liệu đúng (positive tại index $0$) so với các negative candidates trong cùng nhóm với nhiệt độ $\tau = 0.05$:

$$
\mathcal{L}_{\text{NCE}} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(s_{i0} / \tau)}{\sum_{j=0}^{K-1} \exp(s_{ij} / \tau)}
$$

Trong đó:
- $B$: Kích thước batch (số nhóm query - documents).
- $K = 4$: Số lượng tài liệu trong mỗi nhóm ($1$ positive + $3$ negatives).
- $s_{ij} = \cos(f_S(q_i), f_S(d_{ij}))$: Điểm tương đồng cosine giữa query $q_i$ và document $d_{ij}$ tạo bởi mô hình Student.
- $\tau = 0.05$: Hệ số nhiệt độ (temperature).

#### 2. Relational KD Loss ($\mathcal{L}_{\text{RKD}}$)
Đo sai số bình phương trung bình (MSE) giữa ma trận Gram của Student ($G_S$) và Teacher ($G_T$) trên **6 cặp quan hệ tam giác trên** (loại bỏ đường chéo chính $j = k$ vì luôn bằng $1$ do đã chuẩn hóa $L_2$):

$$
\mathcal{L}_{\text{RKD}} = \frac{1}{6B} \sum_{i=1}^B \sum_{0 \le j < k < 4} \left( G_{T, jk}^{(i)} - G_{S, jk}^{(i)} \right)^2
$$

Tổng quát cho nhóm $K$ văn bản với $M = \frac{K(K - 1)}{2}$ cặp quan hệ:

$$
\mathcal{L}_{\text{RKD}} = \frac{2}{B K (K - 1)} \sum_{i=1}^B \sum_{0 \le j < k < K} \left( G_{T, jk}^{(i)} - G_{S, jk}^{(i)} \right)^2
$$

#### 3. Hai nhánh thí nghiệm
- **`E1` (Baseline):** Đặt $\lambda_{\text{rel}} = 0.0$ (chỉ huấn luyện InfoNCE giám sát).
- **`E2` (Relational KD):** Đặt $\lambda_{\text{rel}} = 0.5$ (kết hợp InfoNCE + cấu trúc hình học của Teacher).

---

## 2. Cấu trúc thư mục dự án

```text
Embedding-Distillation/
├── README.md                      # Hướng dẫn chi tiết dự án
├── requirements.txt               # Thư viện phụ thuộc
├── configs/                       # Cấu hình siêu tham số
│   ├── e1_infonce.yaml            # Cấu hình cho nhánh E1 (InfoNCE baseline)
│   └── e2_relational_kd.yaml      # Cấu hình cho nhánh E2 (Relational KD)
├── data/                          # Dữ liệu dạng JSONL
│   ├── groups.jsonl               # Nhóm K=4 văn bản (1 pos + 3 negs)
│   └── groups_scored.jsonl        # Dữ liệu đã chấm teacher_gram (4x4)
├── src/                           # Toàn bộ mã nguồn cốt lõi (ngắn gọn, trực quan)
│   ├── __init__.py
│   ├── prepare_data.py            # Chuẩn bị dữ liệu K=4 (hỗ trợ streaming & demo)
│   ├── score_teacher.py           # Chạy Teacher offline tính Gram matrix
│   ├── train.py                   # Huấn luyện Student LoRA (PyTorch chuẩn)
│   └── evaluate.py                # Đánh giá retrieval cục bộ & MTEB
├── tests/                         # Thư mục kiểm thử toán học
│   └── test_rkd.py                # Unit test kiểm tra Gram, RKD loss, InfoNCE
├── scripts/                       # Scripts chạy tự động hóa 1 click
│   ├── run_pipeline.bat           # Dành cho Windows
│   └── run_pipeline.sh            # Dành cho Linux / GPU Server
└── docs/                          # Tài liệu kỹ thuật chi tiết
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

## 4. Hướng dẫn thực thi từng bước

### Bước 1: Chuẩn bị dữ liệu (`src/prepare_data.py`)
- **Tạo nhanh 50 nhóm mẫu tiếng Việt để test quy trình:**
  ```bash
  python src/prepare_data.py --demo --n 50 --out data/groups.jsonl
  ```
- **Tải và stream dữ liệu thật từ Hugging Face (20.000 nhóm mMARCO):**
  ```bash
  python src/prepare_data.py --n 20000 --out data/groups.jsonl --seed 42
  ```

---

### Bước 2: Chạy Teacher Offline tính Gram Matrix (`src/score_teacher.py`)
- **Chạy thực tế với GPU bằng Teacher `Qwen/Qwen3-Embedding-4B`:**
  ```bash
  python src/score_teacher.py \
    --input data/groups.jsonl \
    --output data/groups_scored.jsonl \
    --model Qwen/Qwen3-Embedding-4B \
    --batch-groups 8
  ```
- **Chạy chế độ mock để test kiểm tra pipeline (không cần tải weights 4B):**
  ```bash
  python src/score_teacher.py --input data/groups.jsonl --output data/groups_scored.jsonl --mock
  ```

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
- **Đánh giá nhanh trên tập cục bộ (Top-1, Top-3, MRR, nDCG@10):**
  ```bash
  python src/evaluate.py --model outputs/E2_RKD --eval-file data/groups_scored.jsonl
  ```
- **Đánh giá trên chuẩn VN-MTEB (Retrieval tasks):**
  ```bash
  python src/evaluate.py --model outputs/E2_RKD --mteb --output results/E2_MTEB
  ```
