@echo off
chcp 65001 >nul
echo ==============================================================
echo   VIETNAMESE RELATIONAL KNOWLEDGE DISTILLATION (QWEN3)
echo ==============================================================

cd /d "%~dp0\.."

echo [1/4] Chạy Unit Test kiểm tra tính toán toán học...
python tests\test_rkd.py
if %ERRORLEVEL% neq 0 (
    echo [LỖI] Unit test thất bại!
    exit /b %ERRORLEVEL%
)

echo.
echo [2/4] Chuẩn bị dữ liệu mẫu tiếng Việt (Demo 50 nhóm)...
python src\prepare_data.py --demo --n 50 --out data\groups.jsonl

echo.
echo [3/4] Chạy Teacher Offline (Mock scoring để test pipeline)...
python src\score_teacher.py --input data\groups.jsonl --output data\groups_scored.jsonl --mock

echo.
echo [4/4] Sẵn sàng huấn luyện!
echo Để huấn luyện E1 (Baseline InfoNCE):
echo   python src\train.py --data data\groups_scored.jsonl --lambda-rel 0.0 --output outputs\E1_InfoNCE
echo.
echo Để huấn luyện E2 (Relational KD):
echo   python src\train.py --data data\groups_scored.jsonl --lambda-rel 0.5 --output outputs\E2_RKD
echo.
echo Để đánh giá mô hình:
echo   python src\evaluate.py --model outputs\E2_RKD --eval-file data\groups_scored.jsonl
echo ==============================================================
pause
