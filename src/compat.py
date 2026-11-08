"""
src/compat.py
Tập trung toàn bộ cấu hình môi trường và bản vá tương thích:
1. Encoding UTF-8 cho Windows console.
2. Bản vá lỗi tương thích giữa PyTorch và Transformers (AuxRequest trong flex_attention).
3. Thiết lập kiểu dữ liệu tối ưu phần cứng (bfloat16 / float16 / TF32).
4. Template định dạng prompt chuẩn cho Qwen3-Embedding.
"""

import sys
import torch

# 1. Thiết lập encoding UTF-8 console Windows
if hasattr(sys.stdout, "reconfigure") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 2. Vá lỗi tương thích giữa PyTorch và Transformers nếu thiếu AuxRequest trong flex_attention
try:
    import torch.nn.attention.flex_attention as _fa
    if not hasattr(_fa, "AuxRequest"):
        _fa.AuxRequest = object
except Exception:
    pass

# 3. Prompt chuẩn cho Qwen3-Embedding
QWEN_TASK_PROMPT = "Given a web search query, retrieve relevant passages that answer the query"


def format_query(query_text: str, is_qwen: bool = True) -> str:
    """Format query theo chuẩn của Qwen3-Embedding."""
    if is_qwen:
        return f"Instruct: {QWEN_TASK_PROMPT}\nQuery: {query_text}"
    return query_text


def get_model_kwargs(device: str = "cuda", is_qwen: bool = True) -> tuple[dict, dict]:
    """
    Trả về (model_kwargs, tokenizer_kwargs) tối ưu theo phần cứng.
    - model_kwargs: tự động chọn bfloat16 (nếu hỗ trợ) hoặc float16, bật TF32.
    - tokenizer_kwargs: padding_side='left' cho Qwen.
    """
    model_kwargs = {}
    if device == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        is_bf16 = torch.cuda.is_bf16_supported()
        model_kwargs["torch_dtype"] = torch.bfloat16 if is_bf16 else torch.float16

    tokenizer_kwargs = {"padding_side": "left"} if is_qwen else {}
    return model_kwargs, tokenizer_kwargs
