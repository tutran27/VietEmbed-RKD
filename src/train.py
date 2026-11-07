"""
train_relational_kd.py
Mục đích: Huấn luyện Student (Qwen/Qwen3-Embedding-0.6B) với hai chế độ:
  - E1: Baseline Supervised Contrastive (InfoNCE) -> thiết lập --lambda-rel 0.0
  - E2: Relational Knowledge Distillation (InfoNCE + Gram RKD) -> thiết lập --lambda-rel 0.5 (hoặc > 0)
"""

import argparse
import json
import os
import random
import sys

# Thiết lập encoding UTF-8 an toàn cho console Windows
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Tương thích giữa torch và transformers nếu thiếu AuxRequest
try:
    import torch.nn.attention.flex_attention as _fa
    if not hasattr(_fa, "AuxRequest"):
        _fa.AuxRequest = object
except Exception:
    pass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer


# Template truy vấn chuẩn theo Qwen3-Embedding
QWEN_TASK_PROMPT = "Given a web search query, retrieve relevant passages that answer the query"


def format_query(query_text: str) -> str:
    """Format query theo chuẩn của Qwen3-Embedding."""
    return f"Instruct: {QWEN_TASK_PROMPT}\nQuery: {query_text}"


class GroupedRetrievalDataset(Dataset):
    """
    Dataset nạp file groups_scored.jsonl.
    Mỗi sample gồm 1 query, 4 documents (index 0 là positive), và ma trận teacher_gram (4x4).
    """
    def __init__(self, jsonl_path: str):
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {jsonl_path}")

        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if len(row["documents"]) != 4:
                    raise ValueError(f"Dòng {line_idx}: Cần đúng 4 documents.")
                self.rows.append(row)

        print(f"--> Đã nạp {len(self.rows)} nhóm dữ liệu từ {jsonl_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate_groups(batch):
    """
    Gom nhóm dữ liệu thành tensor:
    - B queries (được định dạng prompt)
    - B * 4 documents
    - Target teacher Gram shape: [B, 4, 4]
    """
    B = len(batch)
    formatted_queries = [format_query(r["query"]) for r in batch]
    documents = [doc for r in batch for doc in r["documents"]]

    # Kiểm tra xem có teacher_gram không
    has_teacher_gram = "teacher_gram" in batch[0]
    if has_teacher_gram:
        targets = torch.tensor([r["teacher_gram"] for r in batch], dtype=torch.float32)
    else:
        # Nếu chưa có (ví dụ khi chỉ chạy E1 thuần túy mà chưa score teacher)
        targets = torch.zeros((B, 4, 4), dtype=torch.float32)

    return formatted_queries + documents, targets, B


def setup_lora_if_available(model, r=16, alpha=32, dropout=0.05):
    """Gắn LoRA adapter vào base model để huấn luyện tiết kiệm VRAM."""
    try:
        from peft import LoraConfig, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model.add_adapter(lora_config)
        print("--> Đã kích hoạt LoRA adapter thành công.")
        return True
    except ImportError:
        print("--> CẢNH BÁO: Thư viện `peft` chưa được cài đặt. Tiếp tục huấn luyện không dùng LoRA...")
        return False
    except Exception as e:
        print(f"--> CẢNH BÁO: Không thể áp dụng LoRA ({e}). Huấn luyện bình thường...")
        return False


def train():
    parser = argparse.ArgumentParser(description="Huấn luyện Student với Relational Knowledge Distillation / InfoNCE")
    parser.add_argument("--data", default="data/groups_scored.jsonl", help="Đường dẫn file dữ liệu đã tính teacher_gram")
    parser.add_argument("--output", default="outputs/E2_RKD", help="Thư mục lưu mô hình sau huấn luyện")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B", help="Model Student khởi đầu")
    parser.add_argument("--lambda-rel", type=float, default=0.5, help="Hệ số RKD (0.0: E1 Baseline InfoNCE, >0: E2 RKD)")
    parser.add_argument("--batch-groups", type=int, default=4, help="Số group mỗi microbatch (B)")
    parser.add_argument("--accum", type=int, default=8, help="Số bước tích lũy gradient (Gradient Accumulation)")
    parser.add_argument("--epochs", type=int, default=1, help="Số epoch huấn luyện")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=0.05, help="Nhiệt độ InfoNCE tau")
    parser.add_argument("--max-length", type=int, default=512, help="Độ dài tối đa của sequence (tokens)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-lora", action="store_true", help="Không sử dụng LoRA")
    args = parser.parse_args()

    # Thiết lập seed để tái lập kết quả
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--> Chạy huấn luyện trên: {device}")
    print(f"    Thí nghiệm: {'E1 (InfoNCE Baseline)' if args.lambda_rel == 0.0 else f'E2 (Relational KD, lambda={args.lambda_rel})'}")

    # Khởi tạo Student model
    model_kwargs = {}
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        is_bf16_supported = torch.cuda.is_bf16_supported()
        model_kwargs["torch_dtype"] = torch.bfloat16 if is_bf16_supported else torch.float16

    print(f"--> Khởi tạo Student Model: {args.model}")
    model = SentenceTransformer(
        args.model,
        device=device,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"padding_side": "left"} if "qwen" in args.model.lower() else {}
    )
    model.max_seq_length = args.max_length

    # Áp dụng LoRA
    if not args.no_lora:
        setup_lora_if_available(model)

    model.train()

    # Thu thập các tham số cần update gradient
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        print("--> CẢNH BÁO: Tất cả tham số đang bị freeze, unfreeze toàn bộ để train...")
        for p in model.parameters():
            p.requires_grad = True
        trainable_params = [p for p in model.parameters() if p.requires_grad]

    print(f"--> Số lượng tham số cần huấn luyện: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    dataset = GroupedRetrievalDataset(args.data)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_groups,
        shuffle=True,
        collate_fn=collate_groups,
        drop_last=False
    )

    # Chỉ số lấy tam giác trên (off-diagonal) cho ma trận 4x4 (gồm đúng 6 cặp quan hệ)
    tri_upper = torch.triu_indices(4, 4, offset=1, device=device)

    total_steps = len(loader) * args.epochs
    print(f"--> Bắt đầu huấn luyện: {args.epochs} epoch, tổng {len(loader)} batches/epoch, accum={args.accum}...")

    step_counter = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_nce = 0.0
        epoch_rel = 0.0

        for step, (texts, teacher_gram_target, B) in enumerate(loader):
            step_counter += 1

            # 1. Tokenize và forward qua model: texts = [B queries, B*4 documents]
            features = model.tokenize(texts)
            features = {k: v.to(device) for k, v in features.items()}

            out = model(features)
            embeddings = out["sentence_embedding"]

            # L2 normalize embeddings
            embeddings = F.normalize(embeddings.float(), p=2, dim=-1)

            # Phân tách Queries và Documents
            queries = embeddings[:B]                 # Shape: [B, D]
            docs = embeddings[B:].reshape(B, 4, -1)   # Shape: [B, 4, D]

            # 2. Tính InfoNCE Loss (giám sát nhãn thật, index 0 là positive)
            # Dot product (cosine) giữa query và 4 docs trong nhóm
            logits = torch.einsum("bd,bkd->bk", queries, docs) / args.temperature
            labels = torch.zeros(B, device=device, dtype=torch.long)
            loss_nce = F.cross_entropy(logits, labels)

            # 3. Tính Relational KD Loss (nếu lambda_rel > 0)
            if args.lambda_rel > 0:
                # Student Gram matrix: [B, 4, 4]
                student_gram = torch.bmm(docs, docs.transpose(1, 2))
                t_gram = teacher_gram_target.to(device)

                # Chỉ tính sai số MSE trên 6 phần tử tam giác trên (bỏ qua đường chéo = 1)
                s_vals = student_gram[:, tri_upper[0], tri_upper[1]]
                t_vals = t_gram[:, tri_upper[0], tri_upper[1]]
                loss_rel = F.mse_loss(s_vals, t_vals)
            else:
                loss_rel = torch.tensor(0.0, device=device)

            # Tổng hợp loss
            loss = loss_nce + args.lambda_rel * loss_rel

            # Backward với gradient accumulation
            (loss / args.accum).backward()

            epoch_loss += loss.item()
            epoch_nce += loss_nce.item()
            epoch_rel += loss_rel.item()

            # Optimizer Step khi đủ accum hoặc cuối epoch
            is_update_step = (step + 1) % args.accum == 0 or (step + 1) == len(loader)
            if is_update_step:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if (step + 1) % max(1, len(loader) // 5) == 0 or (step + 1) == len(loader):
                print(f"Epoch [{epoch+1}/{args.epochs}] Step [{step+1}/{len(loader)}] "
                      f"- Loss: {loss.item():.4f} (NCE: {loss_nce.item():.4f}, RKD: {loss_rel.item():.4f})")

    # Lưu checkpoint kết quả
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    
    # Lưu config huấn luyện để phục vụ việc tái lập kết quả
    config_record = {
        "model": args.model,
        "lambda_rel": args.lambda_rel,
        "batch_groups": args.batch_groups,
        "accum": args.accum,
        "epochs": args.epochs,
        "lr": args.lr,
        "temperature": args.temperature,
        "seed": args.seed,
    }
    with open(os.path.join(args.output, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2, ensure_ascii=False)

    print(f"--> Huấn luyện hoàn tất! Checkpoint và cấu hình đã lưu tại: {args.output}")


if __name__ == "__main__":
    train()
