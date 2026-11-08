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

# Đảm bảo Python nhận diện được package src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.compat import format_query

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


def evaluate_mteb(model_path: str, output_dir: str = "results", suite: str = "vietnamese", tasks_subset: list = None):
    """
    Chạy đánh giá chính thức qua MTEB (Massive Text Embedding Benchmark)
    theo các suite: 'vietnamese', 'english', 'multilingual', hoặc 'all'.
    """
    try:
        import mteb
    except ImportError:
        print("Lỗi: Chưa cài đặt thư viện mteb. Hãy chạy: pip install mteb")
        return

    print(f"\n=======================================================")
    print(f"--> [MTEB] Khởi tạo Benchmark cho mô hình: {model_path}")
    print(f"    Suite: {suite.upper()}")
    print(f"=======================================================")

    is_qwen = "qwen" in model_path.lower()
    raw_model = SentenceTransformer(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
    raw_model.eval()

    # Gắn wrapper prompt trực tiếp vào SentenceTransformer mà không làm mất kiểu class
    if is_qwen:
        orig_encode = raw_model.encode
        def qwen_encode(sentences, prompt_name=None, **kwargs):
            if prompt_name == "query":
                if isinstance(sentences, list):
                    sentences = [format_query(s, is_qwen=True) if not str(s).startswith("Instruct:") else s for s in sentences]
                elif isinstance(sentences, str) and not str(sentences).startswith("Instruct:"):
                    sentences = format_query(sentences, is_qwen=True)
            return orig_encode(sentences, **kwargs)
        raw_model.encode = qwen_encode

    # Lựa chọn tasks chuẩn tên quốc tế & VN-MTEB
    if tasks_subset:
        task_names = tasks_subset
    elif suite == "vietnamese":
        task_names = ["VieQuADRetrieval", "SciFact-VN", "ArguAna-VN"]
    elif suite == "english":
        task_names = ["SciFact", "NFCorpus", "ArguAna"]
    elif suite == "multilingual":
        task_names = ["VieQuADRetrieval", "SciFact-VN", "SciFact", "NFCorpus"]
    elif suite == "all":
        task_names = ["VieQuADRetrieval", "SciFact-VN", "ArguAna-VN", "SciFact", "NFCorpus", "ArguAna"]
    else:
        task_names = ["VieQuADRetrieval"]

    selected_tasks = []
    for t_name in task_names:
        try:
            selected_tasks.append(mteb.get_task(t_name))
        except Exception as e:
            print(f"[CẢNH BÁO] Không tải được task '{t_name}': {e}")

    if not selected_tasks:
        print("[LỖI] Không có task MTEB nào sẵn sàng để chạy.")
        return

    out_folder = os.path.join(output_dir, f"{model_path.replace('/', '_')}_{suite}")
    os.makedirs(out_folder, exist_ok=True)
    print(f"--> Bắt đầu đánh giá trên {len(selected_tasks)} tasks: {[t.metadata.name for t in selected_tasks]}...")

    cache = mteb.ResultCache(cache_path=out_folder)
    results = mteb.evaluate(raw_model, tasks=selected_tasks, cache=cache)
    print(f"\n--> [MTEB] Đã hoàn tất đánh giá ({suite.upper()})! Kết quả lưu tại: {out_folder}\n")
    return out_folder


def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình Embedding (Local / MTEB)")
    parser.add_argument("--model", required=True, help="Đường dẫn hoặc tên mô hình cần đánh giá")
    parser.add_argument("--eval-file", default="data/groups_scored.jsonl", help="File dữ liệu kiểm tra cục bộ")
    parser.add_argument("--mteb", action="store_true", help="Chạy đánh giá qua MTEB benchmark")
    parser.add_argument("--suite", default="vietnamese", choices=["vietnamese", "english", "multilingual", "all", "local"], help="Bộ benchmark cần chạy")
    parser.add_argument("--output", default="results", help="Thư mục lưu kết quả MTEB")
    parser.add_argument("--tasks", nargs="+", default=None, help="Danh sách task MTEB tùy chọn")
    args = parser.parse_args()

    if args.mteb or args.suite in ["vietnamese", "english", "multilingual", "all"]:
        evaluate_mteb(args.model, output_dir=args.output, suite=args.suite, tasks_subset=args.tasks)
    else:
        evaluate_local(args.model, data_path=args.eval_file)


if __name__ == "__main__":
    main()
