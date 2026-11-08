"""
src/prepare_data.py
Mục đích: Tải dữ liệu thật từ Hugging Face và gom nhóm K=4 (1 positive + 3 negatives).
Nguồn dữ liệu:
  - Tiếng Việt: minhnguyent546/mmarco-vietnamese-split (triples/train)
  - Tiếng Anh: sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1 (triplet-50)
"""

import argparse
import hashlib
import json
import os
import random
import sys
import unicodedata
from collections import OrderedDict

# Đảm bảo nhận diện package src và UTF-8 console Windows
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def norm_text(x: str) -> str:
    """Chuẩn hóa Unicode NFC và làm sạch khoảng trắng thừa."""
    x = unicodedata.normalize("NFC", str(x or ""))
    return " ".join(x.split()).strip()


def query_positive_id(q: str, p: str) -> str:
    """Tạo mã hash duy nhất cho cặp (query, positive)."""
    return hashlib.sha256((q + "\0" + p).encode("utf-8")).hexdigest()[:16]


def stream_vi_mmarco(n_groups: int = 20000, seed: int = 42):
    """Stream dữ liệu thật tiếng Việt từ mMARCO."""
    from datasets import load_dataset
    print(f"--> Đang kết nối streaming dataset mMARCO tiếng Việt (target: {n_groups} nhóm)...")
    ds = load_dataset(
        "minhnguyent546/mmarco-vietnamese-split",
        "triples",
        split="train",
        streaming=True,
    ).shuffle(seed=seed, buffer_size=10000)

    active = OrderedDict()
    completed = set()
    groups = []

    for row in ds:
        q = norm_text(row.get("query", ""))
        p = norm_text(row.get("positive", ""))
        n = norm_text(row.get("negative", ""))

        if not (q and p and n) or n == p or len(q) < 6:
            continue

        key = query_positive_id(q, p)
        if key in completed:
            continue

        if key not in active:
            active[key] = {
                "id": f"vi_{key}",
                "query": q,
                "documents": [p],
                "source": "mmarco_vi"
            }

        group = active[key]
        if n not in group["documents"]:
            group["documents"].append(n)

        if len(group["documents"]) == 4:
            groups.append(group)
            completed.add(key)
            del active[key]

            if len(groups) % 1000 == 0:
                print(f"    [Tiếng Việt] Đã gom: {len(groups)}/{n_groups} nhóm...")

            if len(groups) >= n_groups:
                break

        if len(active) > 100000:
            active.popitem(last=False)

    return groups


def stream_en_msmarco(n_groups: int = 50000, seed: int = 42):
    """Stream dữ liệu thật tiếng Anh từ MS MARCO triplet-50."""
    from datasets import load_dataset
    print(f"--> Đang kết nối streaming dataset MS MARCO tiếng Anh (target: {n_groups} nhóm)...")
    try:
        ds = load_dataset(
            "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
            "triplet-50",
            split="train",
            streaming=True,
        )

        groups = []
        for idx, row in enumerate(ds):
            q = norm_text(row.get("query", ""))
            p = norm_text(row.get("positive", ""))
            n1 = norm_text(row.get("negative_1", ""))
            n2 = norm_text(row.get("negative_2", ""))
            n3 = norm_text(row.get("negative_3", ""))

            if not (q and p and n1 and n2 and n3) or len(q) < 6:
                continue

            groups.append({
                "id": f"en_msmarco_{idx+1:06d}",
                "query": q,
                "documents": [p, n1, n2, n3],
                "source": "msmarco_en"
            })

            if len(groups) % 2000 == 0:
                print(f"    [Tiếng Anh] Đã gom: {len(groups)}/{n_groups} nhóm...")

            if len(groups) >= n_groups:
                break
        return groups
    except Exception as e:
        print(f"--> [CẢNH BÁO] Kết nối tiếng Anh bị gián đoạn ({e}), dùng {len(groups)} nhóm hiện có.")
        return groups
    finally:
        import gc
        if "ds" in locals():
            del ds
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu nhóm K=4 từ Hugging Face")
    parser.add_argument("--out", default="data/groups.jsonl", help="Đường dẫn file đầu ra")
    parser.add_argument("--n", type=int, default=20000, help="Số nhóm tiếng Việt cần lấy")
    parser.add_argument("--en-samples", type=int, default=0, help="Số nhóm tiếng Anh muốn trộn thêm (mặc định 0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    all_groups = []

    if args.n > 0:
        all_groups.extend(stream_vi_mmarco(n_groups=args.n, seed=args.seed))

    if args.en_samples > 0:
        all_groups.extend(stream_en_msmarco(n_groups=args.en_samples, seed=args.seed))

    random.seed(args.seed)
    random.shuffle(all_groups)

    with open(args.out, "w", encoding="utf-8") as f:
        for g in all_groups:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    print(f"--> Hoàn thành! Đã lưu {len(all_groups)} nhóm vào: {args.out}")


if __name__ == "__main__":
    main()
