"""
score_teacher.py
Mục đích: Chạy Teacher model (Qwen/Qwen3-Embedding-4B) OFFLINE để tính trước ma trận
Gram tương đồng giữa các documents trong mỗi nhóm (K=4), sau đó lưu vào file groups_scored.jsonl.
Lợi ích: Chỉ chạy Teacher 1 lần, giải phóng VRAM GPU trước khi huấn luyện Student.
"""

import argparse
import json
import os
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
from tqdm import tqdm



def read_batches(filename: str, batch_size: int):
    """Đọc file jsonl theo từng batch các group để tối ưu bộ nhớ."""
    batch = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if len(row["documents"]) != 4:
                raise ValueError(f"Mỗi group bắt buộc phải có đúng K=4 documents (ID: {row.get('id')})")
            batch.append(row)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def mock_teacher_scoring(input_path: str, output_path: str):
    """
    Tạo ma trận Gram giả lập (hợp lệ về mặt toán học) để test nhanh pipeline
    khi máy chưa tải weights Teacher 4B.
    """
    print("--> [MOCK MODE] Đang tính toán Teacher Gram giả lập cho testing...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    count = 0
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            if not line.strip():
                continue
            row = json.loads(line)
            # Sinh ngẫu nhiên vector 4 docs, chuẩn hóa L2 rồi tính Gram để đảm bảo đối xứng và diag=1
            vecs = np.random.randn(4, 128).astype(np.float32)
            vecs = vecs / np.linalg.norm(vecs, axis=-1, keepdims=True)
            gram = vecs @ vecs.T
            row["teacher_gram"] = gram.round(6).tolist()
            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(f"--> [MOCK MODE] Hoàn thành! Đã tính Gram cho {count} nhóm -> {output_path}")


def run_teacher_scoring(
    input_path: str,
    output_path: str,
    model_name: str = "Qwen/Qwen3-Embedding-4B",
    batch_groups: int = 8,
    max_length: int = 512,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    from sentence_transformers import SentenceTransformer

    print(f"--> Khởi tạo Teacher Model: {model_name} trên thiết bị: {device}")
    
    # Thiết lập dtype phù hợp với phần cứng
    model_kwargs = {}
    if device == "cuda":
        # Ưu tiên bfloat16 nếu GPU hỗ trợ, nếu không thì float16
        is_bf16_supported = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if is_bf16_supported else torch.float16
        model_kwargs = {"torch_dtype": dtype}
        print(f"    Sử dụng GPU dtype: {dtype}")

    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"padding_side": "left"} if "qwen" in model_name.lower() else {}
    )
    model.max_seq_length = max_length
    model.eval()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    total_groups = sum(1 for line in open(input_path, "r", encoding="utf-8") if line.strip())
    total_batches = (total_groups + batch_groups - 1) // batch_groups

    print(f"--> Bắt đầu tính Gram matrix cho {total_groups} nhóm (batch={batch_groups})...")

    with open(output_path, "w", encoding="utf-8") as out:
        for group_batch in tqdm(read_batches(input_path, batch_groups), total=total_batches, desc="Scoring Teacher"):
            # Thu thập toàn bộ documents trong batch (mỗi group có 4 docs -> tổng B*4 docs)
            docs = [doc for row in group_batch for doc in row["documents"]]

            with torch.inference_mode():
                # Encode với L2 normalization tự động
                embeddings = model.encode(
                    docs,
                    batch_size=min(32, len(docs)),
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )

            b = len(group_batch)
            # Reshape về [B, 4, D]
            z = np.asarray(embeddings, dtype=np.float32).reshape(b, 4, -1)
            
            # Tính Gram matrix: G = Z @ Z.T -> Shape [B, 4, 4]
            gram = z @ np.swapaxes(z, 1, 2)

            # Sanity checks: không chứa NaN hoặc Inf
            if not np.isfinite(gram).all():
                raise ValueError("LỖI: Ma trận Gram chứa giá trị NaN hoặc Inf!")

            # Lưu vào từng row
            for i, row in enumerate(group_batch):
                row["teacher_gram"] = gram[i].round(6).tolist()
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"--> Hoàn thành tính Gram matrix Teacher! File lưu tại: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Tính trước Gram matrix từ Teacher Model (offline cache)")
    parser.add_argument("--input", default="data/groups.jsonl", help="Đường dẫn file groups đầu vào")
    parser.add_argument("--output", default="data/groups_scored.jsonl", help="Đường dẫn file lưu kết quả có teacher_gram")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-4B", help="Tên hoặc đường dẫn Teacher model")
    parser.add_argument("--batch-groups", type=int, default=8, help="Số nhóm xử lý trong mỗi batch")
    parser.add_argument("--max-length", type=int, default=512, help="Độ dài tối đa của văn bản (tokens)")
    parser.add_argument("--mock", action="store_true", help="Chạy chế độ mock để test pipeline nhanh")
    args = parser.parse_args()

    if args.mock:
        mock_teacher_scoring(args.input, args.output)
    else:
        run_teacher_scoring(
            input_path=args.input,
            output_path=args.output,
            model_name=args.model,
            batch_groups=args.batch_groups,
            max_length=args.max_length,
        )


if __name__ == "__main__":
    main()
