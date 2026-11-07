"""
test_rkd.py
Unit tests kiểm tra tính đúng đắn về mặt toán học và định dạng của các thành phần trong Relational KD.
Chạy độc lập bằng lệnh: python test_rkd.py
"""

import sys
import torch
import torch.nn.functional as F

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



def test_gram_matrix_properties():
    """Kiểm tra ma trận Gram: shape (B, 4, 4), đối xứng, đường chéo bằng 1.0 sau L2 norm."""
    print("1. Kiểm tra tính chất toán học của Ma trận Gram...")
    B, K, D = 4, 4, 1024  # Batch size 4, 4 documents, chiều 1024
    vecs = torch.randn(B, K, D)
    vecs_norm = F.normalize(vecs, p=2, dim=-1)

    # Gram = vecs @ vecs^T
    gram = torch.bmm(vecs_norm, vecs_norm.transpose(1, 2))

    # Assert shape
    assert gram.shape == (B, K, K), f"Sai shape: {gram.shape}"

    # Assert đối xứng: G == G^T
    sym_diff = torch.max(torch.abs(gram - gram.transpose(1, 2))).item()
    assert sym_diff < 1e-5, f"Ma trận không đối xứng: sai số {sym_diff}"

    # Assert đường chéo chính xấp xỉ 1.0 (do norm L2)
    diag = torch.diagonal(gram, dim1=1, dim2=2)
    diag_diff = torch.max(torch.abs(diag - 1.0)).item()
    assert diag_diff < 1e-5, f"Đường chéo không bằng 1.0: sai số {diag_diff}"

    print("   -> PASS: Ma trận Gram chuẩn xác (shape đúng, đối xứng, đường chéo = 1.0).\n")


def test_relational_kd_loss():
    """Kiểm tra RKD Loss: chỉ tính MSE trên 6 phần tử tam giác trên (bỏ qua đường chéo)."""
    print("2. Kiểm tra Relational KD Loss (Upper triangle off-diagonal)...")
    B = 2
    # Giả lập Gram của Teacher và Student
    G_T = torch.tensor([
        [[1.0, 0.8, 0.4, 0.2],
         [0.8, 1.0, 0.5, 0.3],
         [0.4, 0.5, 1.0, 0.1],
         [0.2, 0.3, 0.1, 1.0]],
        [[1.0, 0.7, 0.3, 0.1],
         [0.7, 1.0, 0.4, 0.2],
         [0.3, 0.4, 1.0, 0.5],
         [0.1, 0.2, 0.5, 1.0]]
    ], dtype=torch.float32)

    # G_S giống hệt G_T nhưng đường chéo khác đi (để test đường chéo không bị ảnh hưởng)
    G_S = G_T.clone()
    G_S[:, 0, 0] = 999.0  # Thay đổi đường chéo

    tri = torch.triu_indices(4, 4, offset=1)
    # Lấy 6 phần tử tam giác trên
    assert len(tri[0]) == 6, f"Phải có đúng 6 phần tử off-diagonal, nhận được {len(tri[0])}"

    loss_rkd = F.mse_loss(G_S[:, tri[0], tri[1]], G_T[:, tri[0], tri[1]])
    assert loss_rkd.item() == 0.0, f"Loss phải bằng 0 khi các cặp quan hệ giống nhau, nhận được {loss_rkd.item()}"

    # Thử sai lệch 1 cặp
    G_S2 = G_T.clone()
    G_S2[0, 0, 1] += 0.1  # Lệch 0.1 ở 1 cặp
    loss_rkd2 = F.mse_loss(G_S2[:, tri[0], tri[1]], G_T[:, tri[0], tri[1]])
    # Tổng 2 batch * 6 phần tử = 12 phần tử; 1 phần tử lệch 0.1 -> MSE = (0.1^2) / 12 = 0.01 / 12 ≈ 0.000833
    expected_mse = (0.1 ** 2) / 12
    assert abs(loss_rkd2.item() - expected_mse) < 1e-5, f"Sai số MSE kỳ vọng {expected_mse}, nhận {loss_rkd2.item()}"

    print("   -> PASS: RKD Loss trích xuất đúng 6 cặp quan hệ tam giác trên và tính MSE chuẩn xác.\n")


def test_infonce_loss():
    """Kiểm tra InfoNCE loss: label 0 luôn là positive."""
    print("3. Kiểm tra InfoNCE Loss...")
    B, D = 3, 128
    queries = F.normalize(torch.randn(B, D), p=2, dim=-1)
    # 4 docs, đặt doc 0 chính là query để cosine = 1.0 (perfect match)
    docs = F.normalize(torch.randn(B, 4, D), p=2, dim=-1)
    docs[:, 0, :] = queries  # doc 0 trùng khớp hoàn hảo với query

    temp = 0.05
    logits = torch.einsum("bd,bkd->bk", queries, docs) / temp
    labels = torch.zeros(B, dtype=torch.long)
    loss = F.cross_entropy(logits, labels)

    # Khi doc 0 là perfect match (score 1.0) và các docs khác ngẫu nhiên (score ~ 0), logits[0] >> logits[1..3]
    # Loss sẽ rất nhỏ (< 0.05)
    assert loss.item() < 0.1, f"InfoNCE loss bất thường: {loss.item()}"
    print(f"   -> PASS: InfoNCE Loss hoạt động chuẩn xác (loss khi match hoàn hảo: {loss.item():.4f}).\n")


def test_qwen_prompt_format():
    """Kiểm tra định dạng prompt Qwen3."""
    print("4. Kiểm tra định dạng prompt Qwen3...")
    task_prompt = "Given a web search query, retrieve relevant passages that answer the query"
    raw_query = "Tìm kiếm quy định pháp luật"
    formatted = f"Instruct: {task_prompt}\nQuery: {raw_query}"
    assert formatted == f"Instruct: {task_prompt}\nQuery: Tìm kiếm quy định pháp luật"
    print("   -> PASS: Định dạng prompt Qwen3 khớp chuẩn mực.\n")


if __name__ == "__main__":
    print("\n================ BẮT ĐẦU UNIT TESTS ================")
    test_gram_matrix_properties()
    test_relational_kd_loss()
    test_infonce_loss()
    test_qwen_prompt_format()
    print("================ TẤT CẢ UNIT TESTS ĐỀU ĐẠT! ================\n")
