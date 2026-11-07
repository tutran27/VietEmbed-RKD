#!/usr/bin/env bash
set -e

# Chuyển về thư mục gốc của repository
cd "$(dirname "$0")/.."

echo "=============================================================="
echo "  VIETNAMESE RELATIONAL KNOWLEDGE DISTILLATION (QWEN3)"
echo "=============================================================="

echo "[1/4] Chạy Unit Test kiểm tra tính toán toán học..."
python tests/test_rkd.py

echo ""
echo "[2/4] Chuẩn bị dữ liệu mẫu tiếng Việt..."
python src/prepare_data.py --demo --n 50 --out data/groups.jsonl

echo ""
echo "[3/4] Chạy Teacher Offline (Mock test)..."
python src/score_teacher.py --input data/groups.jsonl --output data/groups_scored.jsonl --mock

echo ""
echo "[4/4] Pipeline hoàn tất sẵn sàng chạy!"
echo "Chạy E1 (InfoNCE baseline):"
echo "  python src/train.py --data data/groups_scored.jsonl --lambda-rel 0.0 --output outputs/E1_InfoNCE"
echo "Chạy E2 (Relational KD):"
echo "  python src/train.py --data data/groups_scored.jsonl --lambda-rel 0.5 --output outputs/E2_RKD"
echo "Đánh giá:"
echo "  python src/evaluate.py --model outputs/E2_RKD --eval-file data/groups_scored.jsonl"
