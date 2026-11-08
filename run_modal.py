"""
run_modal.py
File độc lập hoàn toàn (All-in-one) chạy trên Cloud GPU A100 (80GB) của Modal.

Cách sử dụng:
  1. CHẠY XEM LOG TRỰC TIẾP TRONG TERMINAL:
      python -m modal run run_modal.py --step score

  2. CHẠY NGẦM TRÊN CLOUD MODAL (Dùng cờ -d của Modal CLI, tắt máy thoải mái):
      python -m modal run -d run_modal.py --step score

  3. CÁC BƯỚC THỰC HIỆN:
      - Chuẩn bị data:     python -m modal run run_modal.py --step prepare --n 100000 --en-samples 50000
      - Teacher Scoring:   python -m modal run run_modal.py --step score --batch-groups 64
      - Train E2 (RKD):    python -m modal run run_modal.py --step train --experiment E2 --lambda-rel 0.5
      - Train E1 (InfoNCE):python -m modal run run_modal.py --step train --experiment E1 --lambda-rel 0.0
      - Đánh giá mô hình:  python -m modal run run_modal.py --step evaluate

  4. TẢI CHECKPOINT VỀ MÁY CÁ NHÂN:
      python -m modal volume get embedding-kd-volume /outputs/E2_RKD ./outputs/E2_RKD
"""

import os
import modal

# 1. Khởi tạo App & Cloud Storage Volume bền vững
app = modal.App("vietnamese-relational-kd")
volume = modal.Volume.from_name("embedding-kd-volume", create_if_missing=True)

VOL_PATH = "/vol"
DATA_DIR = f"{VOL_PATH}/data"
OUTPUTS_DIR = f"{VOL_PATH}/outputs"

# 2. Định nghĩa môi trường Cloud (Python 3.11 + PyTorch + Dependencies)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "sentence-transformers>=3.0.0",
        "peft>=0.10.0",
        "datasets>=2.19.0",
        "accelerate>=0.28.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "mteb>=1.12.0",
    )
    .env({
        "TRITON_DEFAULT_FP_FUSION": "1",
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    })
)

QWEN_TASK_PROMPT = "Given a web search query, retrieve relevant passages that answer the query"


# ==============================================================================
# FUNCTION 1: TẢI & GOM NHÓM DỮ LIỆU THẬT TỪ HUGGING FACE
# ==============================================================================
@app.function(
    image=image,
    volumes={VOL_PATH: volume},
    timeout=3600,  # 1 giờ
)
def prepare_data(n_vi: int = 20000, en_samples: int = 0, seed: int = 42):
    import hashlib
    import json
    import random
    import unicodedata
    from collections import OrderedDict
    from datasets import load_dataset

    def norm(x):
        return " ".join(unicodedata.normalize("NFC", str(x or "")).split()).strip()

    def uid(q, p):
        return hashlib.sha256((q + "\0" + p).encode("utf-8")).hexdigest()[:16]

    os.makedirs(DATA_DIR, exist_ok=True)
    out_file = f"{DATA_DIR}/groups.jsonl"
    all_groups = []

    # 1. Kéo dữ liệu tiếng Việt từ mMARCO
    if n_vi > 0:
        print(f"--> [MODAL] Streaming {n_vi} nhóm tiếng Việt từ mMARCO...")
        ds_vi = load_dataset(
            "minhnguyent546/mmarco-vietnamese-split",
            "triples",
            split="train",
            streaming=True,
        ).shuffle(seed=seed, buffer_size=10000)

        active = OrderedDict()
        completed = set()

        for row in ds_vi:
            q, p, n = norm(row.get("query")), norm(row.get("positive")), norm(row.get("negative"))
            if not (q and p and n) or n == p or len(q) < 6:
                continue

            key = uid(q, p)
            if key in completed:
                continue

            if key not in active:
                active[key] = {"id": f"vi_{key}", "query": q, "documents": [p], "source": "mmarco_vi"}

            group = active[key]
            if n not in group["documents"]:
                group["documents"].append(n)

            if len(group["documents"]) == 4:
                all_groups.append(group)
                completed.add(key)
                del active[key]
                if len(all_groups) % 2000 == 0:
                    print(f"    [Tiếng Việt] Đã gom {len(all_groups)}/{n_vi} nhóm...")
                if len(all_groups) >= n_vi:
                    break

            if len(active) > 100000:
                active.popitem(last=False)

    # 2. Kéo dữ liệu tiếng Anh từ MS MARCO triplet-50 (nếu có yêu cầu)
    if en_samples > 0:
        print(f"--> [MODAL] Streaming {en_samples} nhóm tiếng Anh từ MS MARCO...")
        try:
            ds_en = load_dataset(
                "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
                "triplet-50",
                split="train",
                streaming=True,
            )

            en_count = 0
            for idx, row in enumerate(ds_en):
                q, p = norm(row.get("query")), norm(row.get("positive"))
                n1, n2, n3 = norm(row.get("negative_1")), norm(row.get("negative_2")), norm(row.get("negative_3"))
                if not (q and p and n1 and n2 and n3) or len(q) < 6:
                    continue

                all_groups.append({
                    "id": f"en_msmarco_{idx+1:06d}",
                    "query": q,
                    "documents": [p, n1, n2, n3],
                    "source": "msmarco_en"
                })
                en_count += 1
                if en_count % 5000 == 0:
                    print(f"    [Tiếng Anh] Đã gom {en_count}/{en_samples} nhóm...")
                if en_count >= en_samples:
                    break
        except Exception as e:
            print(f"--> [CẢNH BÁO] Kết nối tiếng Anh bị gián đoạn ({e}), tiếp tục với {len(all_groups)} nhóm hiện có.")
        finally:
            import gc
            if "ds_en" in locals():
                del ds_en
            gc.collect()

    # Trộn ngẫu nhiên
    random.seed(seed)
    random.shuffle(all_groups)

    with open(out_file, "w", encoding="utf-8") as f:
        for g in all_groups:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    volume.commit()
    print(f"--> [MODAL] Hoàn thành! Đã lưu {len(all_groups)} nhóm vào Volume: {out_file}")
    return out_file


# ==============================================================================
# FUNCTION 2: CHẠY TEACHER 4B OFFLINE TÍNH GRAM TRÊN GPU A100 (80GB)
# ==============================================================================
@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOL_PATH: volume},
    timeout=14400,  # 4 giờ
)
def score_teacher(batch_groups: int = 64, max_length: int = 512):
    import json
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    volume.reload()
    in_file = f"{DATA_DIR}/groups.jsonl"
    out_file = f"{DATA_DIR}/groups_scored.jsonl"

    if not os.path.exists(in_file):
        raise FileNotFoundError(f"Không tìm thấy {in_file}. Hãy chạy step prepare trước!")

    print(f"--> [MODAL A100] Khởi tạo Teacher Qwen/Qwen3-Embedding-4B...")
    torch.backends.cuda.matmul.allow_tf32 = True

    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-4B",
        device="cuda",
        model_kwargs={"torch_dtype": torch.bfloat16},
        tokenizer_kwargs={"padding_side": "left"},
    )
    model.max_seq_length = max_length
    model.eval()

    def read_batches(filename, b_size):
        batch = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    batch.append(json.loads(line))
                    if len(batch) == b_size:
                        yield batch
                        batch = []
        if batch:
            yield batch

    total_groups = sum(1 for line in open(in_file, "r", encoding="utf-8") if line.strip())
    print(f"--> [MODAL A100] Tính ma trận Gram cho {total_groups} nhóm (batch_groups={batch_groups})...")

    with open(out_file, "w", encoding="utf-8") as out:
        for group_batch in tqdm(read_batches(in_file, batch_groups)):
            docs = [doc for r in group_batch for doc in r["documents"]]

            with torch.inference_mode():
                embeddings = model.encode(
                    docs,
                    batch_size=min(128, len(docs)),
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )

            b = len(group_batch)
            z = np.asarray(embeddings, dtype=np.float32).reshape(b, 4, -1)
            gram = z @ np.swapaxes(z, 1, 2)

            for i, row in enumerate(group_batch):
                row["teacher_gram"] = gram[i].round(6).tolist()
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

    volume.commit()
    print(f"--> [MODAL A100] Teacher Scoring hoàn tất! File đã lưu vào Volume: {out_file}")
    return out_file


# ==============================================================================
# FUNCTION 3: HUẤN LUYỆN STUDENT 0.6B LoRA TRÊN GPU A100 (80GB)
# ==============================================================================
@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOL_PATH: volume},
    timeout=14400,  # 4 giờ
)
def train_student(
    experiment: str = "E2",
    lambda_rel: float = 0.5,
    epochs: int = 1,
    batch_groups: int = 4,
    accum: int = 8,
    lr: float = 1e-4,
    seed: int = 42
):
    import json
    import os
    import random
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from sentence_transformers import SentenceTransformer
    from peft import LoraConfig, TaskType
    from tqdm import tqdm

    # Tắt cảnh báo Triton và hỗ trợ HF_TOKEN
    os.environ["TRITON_DEFAULT_FP_FUSION"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    volume.reload()
    data_file = f"{DATA_DIR}/groups_scored.jsonl"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Không tìm thấy {data_file}. Hãy chạy step score trước!")

    out_dir = f"{OUTPUTS_DIR}/{experiment}_{'RKD' if lambda_rel > 0 else 'InfoNCE'}"
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"--> [MODAL A100] Khởi tạo Student Qwen/Qwen3-Embedding-0.6B (SDPA + BF16)...")
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B",
        device="cuda",
        model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"},
        tokenizer_kwargs={"padding_side": "left"},
    )
    model.max_seq_length = 512

    # LoRA config tối ưu (r=32, alpha=64)
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model.add_adapter(lora_config)

    # BẬT GRADIENT CHECKPOINTING ĐỂ TIẾT KIỆM 80% VRAM
    base_model = model[0].auto_model
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
        print("--> [MODAL A100] Đã kích hoạt Gradient Checkpointing.")
    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False

    model.train()

    class RetrievalDataset(Dataset):
        def __init__(self, path):
            self.rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, idx):
            return self.rows[idx]

    def collate_fn(batch):
        B = len(batch)
        queries = [f"Instruct: {QWEN_TASK_PROMPT}\nQuery: {r['query']}" for r in batch]
        docs = [d for r in batch for d in r["documents"]]
        targets = torch.tensor([r.get("teacher_gram", [[0]*4]*4) for r in batch], dtype=torch.float32)
        return queries + docs, targets, B

    loader = DataLoader(
        RetrievalDataset(data_file),
        batch_size=batch_groups,
        shuffle=True,
        collate_fn=collate_fn
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    tri_upper = torch.triu_indices(4, 4, offset=1, device="cuda")

    print(f"--> [MODAL A100] Bắt đầu Train {experiment} (lambda_rel={lambda_rel}, batch_groups={batch_groups}, accum={accum})...", flush=True)
    optimizer.zero_grad(set_to_none=True)

    best_loss = float("inf")
    best_dir = f"{out_dir}/best"

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, (texts, target_gram, B) in enumerate(pbar):
            # Tokenize chuẩn tắc HuggingFace (không deprecation, không lỗi str)
            feats = model.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            embeddings = F.normalize(model(feats)["sentence_embedding"].float(), p=2, dim=-1)

            queries = embeddings[:B]
            docs = embeddings[B:].reshape(B, 4, -1)

            # InfoNCE Loss
            logits = torch.einsum("bd,bkd->bk", queries, docs) / 0.05
            labels = torch.zeros(B, device="cuda", dtype=torch.long)
            loss_nce = F.cross_entropy(logits, labels)

            # Relational KD Loss
            if lambda_rel > 0:
                student_gram = torch.bmm(docs, docs.transpose(1, 2))
                t_gram = target_gram.to("cuda")
                loss_rel = F.mse_loss(
                    student_gram[:, tri_upper[0], tri_upper[1]],
                    t_gram[:, tri_upper[0], tri_upper[1]]
                )
            else:
                loss_rel = torch.tensor(0.0, device="cuda")

            loss = loss_nce + lambda_rel * loss_rel
            (loss / accum).backward()

            if (step + 1) % accum == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            current_loss = loss.item()

            # Cập nhật thanh tiến độ liên tục và in log mỗi 50 steps
            pbar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "nce": f"{loss_nce.item():.4f}",
                "rkd": f"{loss_rel.item():.4f}",
                "best": f"{best_loss:.4f}" if best_loss != float("inf") else "None"
            })

            if (step + 1) % 50 == 0 or (step + 1) == len(loader):
                print(f"Epoch [{epoch+1}/{epochs}] Step [{step+1}/{len(loader)}] - Loss: {current_loss:.4f} (NCE: {loss_nce.item():.4f}, RKD: {loss_rel.item():.4f})", flush=True)

            # LƯU CHECKPOINT TỐT NHẤT: Cứ mỗi 500 steps kiểm tra nếu loss tốt nhất thì lưu ngay vào Volume
            if (step + 1) % 500 == 0 or (step + 1) == len(loader):
                if current_loss < best_loss:
                    best_loss = current_loss
                    os.makedirs(best_dir, exist_ok=True)
                    model.save_pretrained(best_dir)

                    summary = {
                        "experiment": experiment,
                        "lambda_rel": lambda_rel,
                        "best_loss": round(best_loss, 4),
                        "best_step": step + 1,
                        "total_steps": len(loader),
                        "epoch": epoch + 1
                    }
                    with open(f"{best_dir}/best_summary.json", "w", encoding="utf-8") as f:
                        json.dump(summary, f, indent=2)

                    volume.commit()
                    print(f"--> [BEST CHECKPOINT] Đã lưu mô hình tốt nhất mới (Loss={best_loss:.4f}) tại: {best_dir}", flush=True)

    # Lưu thêm checkpoint cuối cùng vào thư mục out_dir
    model.save_pretrained(out_dir)
    volume.commit()
    print(f"--> [MODAL A100] Huấn luyện hoàn tất! Checkpoint cuối lưu tại: {out_dir}", flush=True)
    print(f"--> [MODAL A100] Checkpoint tốt nhất (Best Loss={best_loss:.4f}) lưu tại: {best_dir}", flush=True)
    return best_dir


# ==============================================================================
# FUNCTION 4: ĐÁNH GIÁ MÔ HÌNH TOÀN DIỆN (LOCAL / VIETNAMESE / ENGLISH / MULTILINGUAL)
# ==============================================================================
@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOL_PATH: volume},
    timeout=3600,
)
def evaluate_model(model_path: str, suite: str = "local", tasks: str = ""):
    """
    Đánh giá mô hình theo các bộ benchmark:
      - 'local': Đánh giá nhanh 1.000 mẫu cục bộ (Top-1, Top-3, MRR, nDCG@10) - ~20s
      - 'vietnamese': VN-MTEB (ViQuADRetrieval, SciFact-VN, ArguAna-VN) - ~3 phút
      - 'english': MTEB English (SciFact, NFCorpus, ArguAna) - ~2 phút
      - 'multilingual': MMTEB (mMarcoRetrieval) - ~4 phút
      - 'all': Chạy cả 3 bộ MTEB (Việt + Anh + Đa ngữ) - ~8 phút
    """
    import json
    import math
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    volume.reload()
    print(f"\n==============================================================")
    print(f"--> [MODAL A100] Bắt đầu đánh giá mô hình: {model_path}")
    print(f"    Suite đánh giá: {suite.upper()}")
    print(f"==============================================================")

    is_qwen = "qwen" in model_path.lower()
    model = SentenceTransformer(model_path, device="cuda")
    model.eval()

    if suite == "local":
        eval_file = f"{DATA_DIR}/groups_scored.jsonl"
        if not os.path.exists(eval_file):
            eval_file = f"{DATA_DIR}/groups.jsonl"

        samples = [json.loads(line) for line in open(eval_file, encoding="utf-8") if line.strip()][:1000]

        top1, top3, mrr_sum, ndcg_sum = 0, 0, 0.0, 0.0
        for item in samples:
            q_text = item["query"]
            q = f"Instruct: {QWEN_TASK_PROMPT}\nQuery: {q_text}" if is_qwen else q_text
            docs = item["documents"]

            with torch.inference_mode():
                q_emb = model.encode(q, normalize_embeddings=True, convert_to_numpy=True)
                d_embs = model.encode(docs, normalize_embeddings=True, convert_to_numpy=True)

            scores = np.dot(d_embs, q_emb)
            rank = np.argsort(-scores).tolist().index(0) + 1

            if rank == 1: top1 += 1
            if rank <= 3: top3 += 1
            mrr_sum += 1.0 / rank
            ndcg_sum += 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0

        n = len(samples)
        print(f"\n--- KẾT QUẢ ĐÁNH GIÁ CỤC BỘ ({n} mẫu) ---")
        print(f"  * Model:           {model_path}")
        print(f"  * Top-1 Accuracy:  {top1 / n * 100:.2f}%")
        print(f"  * Top-3 Recall:    {top3 / n * 100:.2f}%")
        print(f"  * MRR:             {mrr_sum / n:.4f}")
        print(f"  * nDCG@10:         {ndcg_sum / n:.4f}")
        print("------------------------------------------\n")
        return {"top1": top1 / n, "top3": top3 / n, "mrr": mrr_sum / n, "ndcg10": ndcg_sum / n}

    # ĐÁNH GIÁ QUA THƯ VIỆN CHUẨN MTEB
    import mteb

    def setup_qwen_prompts(m):
        """Cấu hình prompt chuẩn cho Qwen3-Embedding trong SentenceTransformers."""
        if not hasattr(m, "prompts") or m.prompts is None:
            m.prompts = {}
        m.prompts["query"] = f"Instruct: {QWEN_TASK_PROMPT}\nQuery: "
        m.default_prompt_name = "query"

        if hasattr(m, "encode_query"):
            orig_eq = m.encode_query
            def qwen_eq(*args, **kwargs):
                if "prompt_name" not in kwargs:
                    kwargs["prompt_name"] = "query"
                return orig_eq(*args, **kwargs)
            m.encode_query = qwen_eq

        orig_enc = m.encode
        def qwen_enc(*args, **kwargs):
            p_name = kwargs.get("prompt_name") or kwargs.get("prompt")
            if p_name == "query":
                if len(args) > 0:
                    s = args[0]
                    if isinstance(s, list):
                        s = [f"Instruct: {QWEN_TASK_PROMPT}\nQuery: {x}" if not str(x).startswith("Instruct:") else x for x in s]
                    elif isinstance(s, str) and not s.startswith("Instruct:"):
                        s = f"Instruct: {QWEN_TASK_PROMPT}\nQuery: {s}"
                    args = (s,) + args[1:]
                elif "sentences" in kwargs:
                    s = kwargs["sentences"]
                    if isinstance(s, list):
                        s = [f"Instruct: {QWEN_TASK_PROMPT}\nQuery: {x}" if not str(x).startswith("Instruct:") else x for x in s]
                    elif isinstance(s, str) and not s.startswith("Instruct:"):
                        s = f"Instruct: {QWEN_TASK_PROMPT}\nQuery: {s}"
                    kwargs["sentences"] = s
            return orig_enc(*args, **kwargs)
        m.encode = qwen_enc

    if is_qwen:
        setup_qwen_prompts(model)

    # Lựa chọn danh sách tasks theo Suite (chuẩn tên MTEB quốc tế & VN-MTEB)
    task_names = []
    if tasks:
        task_names = [t.strip() for t in tasks.split(",") if t.strip()]
    elif suite == "vietnamese":
        task_names = ["VieQuADRetrieval", "SciFact-VN", "ArguAna-VN"]
    elif suite == "english":
        task_names = ["SciFact", "NFCorpus", "ArguAna"]
    elif suite == "multilingual":
        task_names = ["VieQuADRetrieval", "SciFact-VN", "SciFact", "NFCorpus"]
    elif suite == "all":
        task_names = ["VieQuADRetrieval", "SciFact-VN", "ArguAna-VN", "SciFact", "NFCorpus", "ArguAna"]

    selected_tasks = []
    for t_name in task_names:
        try:
            t = mteb.get_task(t_name)
            selected_tasks.append(t)
        except Exception as e:
            print(f"[CẢNH BÁO] Không tải được task '{t_name}': {e}")

    if not selected_tasks:
        print("[LỖI] Không có task MTEB nào hợp lệ được tìm thấy!")
        return

    out_res = f"{VOL_PATH}/results/{model_path.replace('/', '_')}_{suite}"
    os.makedirs(out_res, exist_ok=True)
    print(f"--> Chạy MTEB trên {len(selected_tasks)} tasks: {[t.metadata.name for t in selected_tasks]}...")

    cache = mteb.ResultCache(cache_path=out_res)
    results = mteb.evaluate(model, tasks=selected_tasks, cache=cache)
    volume.commit()

    print(f"\n--> [MODAL] Đã hoàn tất đánh giá MTEB ({suite.upper()})! Kết quả lưu tại: {out_res}\n")
    return out_res


# ==============================================================================
# FUNCTION 5: CHẠY LIÊN HOÀN TOÀN BỘ 5 MÔ HÌNH TRÊN MTEB
# ==============================================================================
@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOL_PATH: volume},
    timeout=7200,  # 2 giờ
)
def benchmark_all(suite: str = "vietnamese"):
    """
    Chạy đánh giá liên hoàn 5 mô hình trên các bộ task MTEB:
    - /vol/outputs/E2_RKD/best (Mô hình ta train)
    - Qwen/Qwen3-Embedding-0.6B (Base Student)
    - Qwen/Qwen3-Embedding-4B (Teacher)
    - BAAI/bge-m3 (Tượng đài đa ngôn ngữ)
    - bkai-foundation-models/vietnamese-bi-encoder (Quán quân tiếng Việt)
    """
    import os
    import json
    import gc
    import torch
    import mteb
    from sentence_transformers import SentenceTransformer

    volume.reload()

    models_to_test = [
        ("Student-0.6B-RKD (Best)", "/vol/outputs/E2_RKD/best"),
        ("Qwen3-Embedding-0.6B (Base)", "Qwen/Qwen3-Embedding-0.6B"),
        ("Qwen3-Embedding-4B (Teacher)", "Qwen/Qwen3-Embedding-4B"),
        ("BGE-M3 (Multilingual)", "BAAI/bge-m3"),
        ("BKAI-Bi-Encoder (Vietnamese)", "bkai-foundation-models/vietnamese-bi-encoder"),
    ]

    if suite == "vietnamese":
        task_names = ["VieQuADRetrieval", "SciFact-VN", "ArguAna-VN"]
    elif suite == "english":
        task_names = ["SciFact", "NFCorpus", "ArguAna"]
    else:  # all
        task_names = ["VieQuADRetrieval", "SciFact-VN", "ArguAna-VN", "SciFact", "NFCorpus", "ArguAna"]

    print("==================================================================")
    print(f"--> [MODAL BENCHMARK] Chạy 5 mô hình trên Suite: {suite.upper()}")
    print(f"    Danh sách tasks: {task_names}")
    print("==================================================================")

    selected_tasks = []
    for t_name in task_names:
        try:
            selected_tasks.append(mteb.get_task(t_name))
        except Exception as e:
            print(f"[CẢNH BÁO] Không tải được task '{t_name}': {e}")

    summary_table = []

    for name, path in models_to_test:
        print(f"\n>>> [EVALUATING] {name} ({path})...")
        is_qwen = "qwen" in path.lower() or "rkd" in path.lower()

        # Load model
        model = SentenceTransformer(path, device="cuda")
        model.eval()

        def setup_qwen_prompts(m):
            if not hasattr(m, "prompts") or m.prompts is None:
                m.prompts = {}
            m.prompts["query"] = f"Instruct: {QWEN_TASK_PROMPT}\nQuery: "
            m.default_prompt_name = "query"

            if hasattr(m, "encode_query"):
                orig_eq = m.encode_query
                def qwen_eq(*args, **kwargs):
                    if "prompt_name" not in kwargs:
                        kwargs["prompt_name"] = "query"
                    return orig_eq(*args, **kwargs)
                m.encode_query = qwen_eq

            orig_enc = m.encode
            def qwen_enc(*args, **kwargs):
                p_name = kwargs.get("prompt_name") or kwargs.get("prompt")
                if p_name == "query":
                    if len(args) > 0:
                        s = args[0]
                        if isinstance(s, list):
                            s = [f"Instruct: {QWEN_TASK_PROMPT}\nQuery: {x}" if not str(x).startswith("Instruct:") else x for x in s]
                        elif isinstance(s, str) and not s.startswith("Instruct:"):
                            s = f"Instruct: {QWEN_TASK_PROMPT}\nQuery: {s}"
                        args = (s,) + args[1:]
                    elif "sentences" in kwargs:
                        s = kwargs["sentences"]
                        if isinstance(s, list):
                            s = [f"Instruct: {QWEN_TASK_PROMPT}\nQuery: {x}" if not str(x).startswith("Instruct:") else x for x in s]
                        elif isinstance(s, str) and not s.startswith("Instruct:"):
                            s = f"Instruct: {QWEN_TASK_PROMPT}\nQuery: {s}"
                        kwargs["sentences"] = s
                return orig_enc(*args, **kwargs)
            m.encode = qwen_enc

        if is_qwen:
            setup_qwen_prompts(model)

        out_res = f"{VOL_PATH}/results/{path.replace('/', '_')}_{suite}"
        os.makedirs(out_res, exist_ok=True)
        cache = mteb.ResultCache(cache_path=out_res)

        mteb.evaluate(model, tasks=selected_tasks, cache=cache)
        volume.commit()

        # Đọc điểm kết quả từ cache JSON
        row = {"Model": name}
        scores_list = []
        for t in selected_tasks:
            t_name = t.metadata.name
            task_score = None
            for root, dirs, files in os.walk(out_res):
                if f"{t_name}.json" in files:
                    try:
                        with open(os.path.join(root, f"{t_name}.json"), "r", encoding="utf-8") as f:
                            data = json.load(f)
                            splits = list(data.get("scores", {}).keys())
                            if splits:
                                split_scores = data["scores"][splits[0]]
                                if isinstance(split_scores, list) and len(split_scores) > 0:
                                    task_score = split_scores[0].get("ndcg_at_10")
                    except Exception:
                        pass
            row[t_name] = round(task_score, 4) if task_score is not None else "N/A"
            if task_score is not None:
                scores_list.append(task_score)

        row["Average"] = round(sum(scores_list) / len(scores_list), 4) if scores_list else "N/A"
        summary_table.append(row)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # In bảng tổng hợp đẹp mắt
    print("\n\n" + "=" * 80)
    print(f"               BẢNG TỔNG HỢP KẾT QUẢ MTEB ({suite.upper()})")
    print("=" * 80)
    header = ["Model"] + [t.metadata.name for t in selected_tasks] + ["AVERAGE"]
    print(" | ".join(header))
    print("-" * 80)
    for r in summary_table:
        line = [str(r.get("Model", ""))] + [str(r.get(t.metadata.name, "N/A")) for t in selected_tasks] + [str(r.get("Average", "N/A"))]
        print(" | ".join(line))
    print("=" * 80 + "\n")

    summary_file = f"{VOL_PATH}/results/benchmark_summary_{suite}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_table, f, indent=2, ensure_ascii=False)
    volume.commit()
    print(f"--> Đã lưu bảng xếp hạng hoàn chỉnh tại: {summary_file}")
    return summary_table


# ==============================================================================
# LOCAL ENTRYPOINT: ĐIỀU KHIỂN TỪ TERMINAL MÁY CỦA BẠN
# ==============================================================================
@app.local_entrypoint()
def main(
    step: str = "all",            # all, prepare, score, train, evaluate, benchmark
    experiment: str = "E2",       # E1 hoặc E2
    lambda_rel: float = 0.5,      # 0.0 cho E1, 0.5 cho E2
    n: int = 20000,               # Số nhóm tiếng Việt
    en_samples: int = 0,          # Số nhóm tiếng Anh trộn thêm (mặc định 0)
    batch_groups: int = 64,       # Số nhóm trong 1 batch của Teacher 4B
    epochs: int = 1,              # Số epoch train
    suite: str = "vietnamese",    # local, vietnamese, english, multilingual, all
    eval_model: str = "",         # Tên model cần đánh giá (mặc định lấy model vừa train)
    tasks: str = ""               # Danh sách task tùy chọn (cách nhau bởi dấu phẩy)
):
    print("==============================================================")
    print("   CHẠY VIETNAMESE RELATIONAL KD TRÊN CLOUD GPU A100 (MODAL)")
    print("==============================================================")

    # 1. BƯỚC PREPARE
    if step in ["prepare", "all"]:
        print(f"\n--> [1/3] Chuẩn bị dữ liệu thật (Việt: {n}, Anh: {en_samples})...")
        prepare_data.remote(n_vi=n, en_samples=en_samples)

    # 2. BƯỚC SCORE TEACHER 4B
    if step in ["score", "all"]:
        print(f"\n--> [2/3] Chạy Teacher 4B Offline trên GPU A100 (batch_groups={batch_groups})...")
        score_teacher.remote(batch_groups=batch_groups)

    # 3. BƯỚC TRAIN STUDENT 0.6B
    if step in ["train", "all"]:
        print(f"\n--> [3/3] Huấn luyện Student 0.6B trên A100 ({experiment}, lambda={lambda_rel})...")
        model_dir = train_student.remote(
            experiment=experiment,
            lambda_rel=lambda_rel,
            epochs=epochs,
            batch_groups=16,
            accum=2
        )
        print("\n--> Đánh giá mô hình vừa huấn luyện...")
        evaluate_model.remote(model_path=model_dir, suite=suite, tasks=tasks)

    # 4. BƯỚC CHỈ ĐÁNH GIÁ 1 MODEL
    elif step == "evaluate":
        target = eval_model if eval_model else f"{OUTPUTS_DIR}/{experiment}_{'RKD' if lambda_rel > 0 else 'InfoNCE'}/best"
        evaluate_model.remote(model_path=target, suite=suite, tasks=tasks)

    # 5. BƯỚC BENCHMARK TOÀN DIỆN TẤT CẢ 5 MODEL
    elif step == "benchmark":
        print(f"\n--> Chạy Benchmark toàn diện 5 mô hình trên Suite: {suite.upper()}...")
        benchmark_all.remote(suite=suite)

    print("\n--> TIẾN TRÌNH ĐÃ HOÀN TẤT THÀNH CÔNG!")
