"""
evaluate_models.py
Mục đích: Đánh giá chất lượng của các mô hình Embedding:
  - Teacher 4B, E0 (0.6B gốc), E1 (InfoNCE), E2 (InfoNCE + RKD), và các baseline đối chiếu.
  - Hỗ trợ 2 chế độ:
    1. Đánh giá nhanh trên tập cục bộ (Top-1, Top-3, MRR, nDCG) không cần cài đặt benchmark phức tạp.
    2. Đánh giá chính thức trên VN-MTEB thông qua thư viện `mteb`.
"""

import argparse
import json
import math
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
from sentence_transformers import SentenceTransformer


QWEN_TASK_PROMPT = "Given a web search query, retrieve relevant passages that answer the query"


def format_query(query_text: str, is_qwen: bool = True) -> str:
    """Format query với instruction nếu là model Qwen3-Embedding."""
    if is_qwen:
        return f"Instruct: {QWEN_TASK_PROMPT}\nQuery: {query_text}"
    return query_text


def compute_dcg_at_k(ranks: list, k: int = 10) -> float:
    """Tính DCG@K với ranks là vị trí của positive (1-based index)."""
    dcg = 0.0
    for r in ranks:
        if r <= k:
            dcg += 1.0 / math.log2(r + 1)
    return dcg


def evaluate_local(model_path: str, data_path: str, batch_size: int = 16, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Đánh giá độ chính xác retrieval trên tập dữ liệu cục bộ (như data/groups.jsonl hoặc dev set).
    Index 0 của documents trong mỗi nhóm luôn là positive.
    """
    print(f"\n=======================================================")
    print(f"--> Đang đánh giá cục bộ mô hình: {model_path}")
    print(f"    Tập dữ liệu: {data_path} trên {device}")
    print(f"=======================================================")

    is_qwen = "qwen" in model_path.lower()
    model = SentenceTransformer(model_path, device=device)
    model.eval()

    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if not samples:
        print("Tập dữ liệu rỗng!")
        return

    top1_correct = 0
    top3_correct = 0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    total = len(samples)

    for item in samples:
        q_raw = item["query"]
        docs = item["documents"]  # docs[0] là positive

        q_formatted = format_query(q_raw, is_qwen=is_qwen)

        with torch.inference_mode():
            q_emb = model.encode(q_formatted, normalize_embeddings=True, convert_to_numpy=True)
            d_embs = model.encode(docs, normalize_embeddings=True, convert_to_numpy=True)

        # Tính cosine similarity (vì đã normalize L2 nên chính là dot product)
        scores = np.dot(d_embs, q_emb)

        # Xếp hạng giảm dần theo score
        ranked_indices = np.argsort(-scores).tolist()
        # Tìm vị trí (rank 1-based) của positive document (index gốc = 0)
        rank_pos = ranked_indices.index(0) + 1

        if rank_pos == 1:
            top1_correct += 1
        if rank_pos <= 3:
            top3_correct += 1

        mrr_sum += 1.0 / rank_pos
        # Với 1 ground truth duy nhất, IDCG@10 = 1.0 / log2(1 + 1) = 1.0
        dcg = 1.0 / math.log2(rank_pos + 1) if rank_pos <= 10 else 0.0
        ndcg_sum += dcg

    top1_acc = top1_correct / total
    top3_recall = top3_correct / total
    mrr = mrr_sum / total
    ndcg10 = ndcg_sum / total

    print(f"\n--- KẾT QUẢ ĐÁNH GIÁ (Số mẫu: {total}) ---")
    print(f"  * Top-1 Accuracy:  {top1_acc * 100:.2f}%")
    print(f"  * Top-3 Recall:    {top3_recall * 100:.2f}%")
    print(f"  * MRR:             {mrr:.4f}")
    print(f"  * nDCG@10:         {ndcg10:.4f}")
    print(f"-----------------------------------------\n")

    return {"top1": top1_acc, "top3": top3_recall, "mrr": mrr, "ndcg10": ndcg10}


def evaluate_mteb(model_path: str, output_dir: str = "results", tasks_subset: list = None):
    """
    Chạy đánh giá chính thức qua MTEB (Massive Text Embedding Benchmark)
    theo chuẩn VN-MTEB (vie, v1).
    """
    try:
        import mteb
    except ImportError:
        print("Lỗi: Chưa cài đặt thư viện mteb. Hãy chạy: pip install mteb")
        return

    print(f"--> Đang khởi tạo MTEB Benchmark cho mô hình: {model_path}...")
    model = SentenceTransformer(model_path, device="cuda" if torch.cuda.is_available() else "cpu")

    try:
        benchmark = mteb.get_benchmark("VN-MTEB (vie, v1)")
        if tasks_subset:
            tasks = [t for t in benchmark.tasks if t.metadata.name in tasks_subset]
        else:
            # Mặc định lấy các retrieval tasks
            tasks = [t for t in benchmark.tasks if getattr(t.metadata, "type", "") == "Retrieval"]
    except Exception as e:
        print(f"Không lấy được VN-MTEB qua get_benchmark ({e}), chuyển sang lấy theo task list cụ thể...")
        # Fallback danh sách task pilot
        tasks = []
        for task_name in (tasks_subset or ["SciFact-VN", "ArguAna-VN", "Quora-VN"]):
            try:
                tasks.append(mteb.get_task(task_name))
            except Exception as te:
                print(f"Không tìm thấy task {task_name}: {te}")

    if not tasks:
        print("Không có task MTEB nào sẵn sàng để chạy.")
        return

    print(f"--> Bắt đầu đánh giá trên {len(tasks)} tasks...")
    os.makedirs(output_dir, exist_ok=True)
    results = mteb.evaluate(
        model,
        tasks=tasks,
        cache=mteb.ResultCache(cache_path=output_dir),
    )
    print(f"--> Đã hoàn tất đánh giá MTEB! Kết quả lưu tại: {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình Embedding (Local / MTEB)")
    parser.add_argument("--model", required=True, help="Đường dẫn hoặc tên mô hình cần đánh giá")
    parser.add_argument("--eval-file", default="data/groups_scored.jsonl", help="File dữ liệu kiểm tra cục bộ")
    parser.add_argument("--mteb", action="store_true", help="Chạy đánh giá qua MTEB benchmark")
    parser.add_argument("--output", default="results", help="Thư mục lưu kết quả MTEB")
    parser.add_argument("--tasks", nargs="+", default=None, help="Danh sách task MTEB cần chạy")
    args = parser.parse_args()

    if args.mteb:
        evaluate_mteb(args.model, output_dir=args.output, tasks_subset=args.tasks)
    else:
        evaluate_local(args.model, data_path=args.eval_file)


if __name__ == "__main__":
    main()
