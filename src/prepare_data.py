"""
prepare_groups.py
Mục đích: Chuẩn bị dữ liệu nhóm K=4 (1 positive + 3 negatives) cho mỗi query.
Output format: JSONL mỗi dòng gồm {"id": ..., "query": ..., "documents": [pos, neg1, neg2, neg3], "source": ...}
"""

import argparse
import hashlib
import json
import os
import sys
import unicodedata
from collections import OrderedDict

# Thiết lập encoding UTF-8 an toàn cho console Windows
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass



def norm_text(x: str) -> str:
    """Chuẩn hóa Unicode NFC và làm sạch khoảng trắng thừa tiếng Việt."""
    x = unicodedata.normalize("NFC", str(x or ""))
    return " ".join(x.split()).strip()


def query_positive_id(q: str, p: str) -> str:
    """Tạo mã hash duy nhất cho cặp (query, positive)."""
    return hashlib.sha256((q + "\0" + p).encode("utf-8")).hexdigest()[:16]


def create_sample_data(out_path: str, n_groups: int = 20):
    """
    Tạo dữ liệu mẫu tiếng Việt cục bộ để chạy thử nghiệm nhanh (smoke test)
    mà không cần tải dataset 40M dòng từ Hugging Face.
    """
    print(f"--> Đang tạo {n_groups} nhóm dữ liệu mẫu tiếng Việt để test...")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    templates = [
        {
            "query": "Thủ tục xin cấp lại căn cước công dân gắn chip như thế nào?",
            "pos": "Để xin cấp lại thẻ căn cước công dân, công dân có thể nộp hồ sơ trực tuyến qua Cổng dịch vụ công hoặc đến trực tiếp cơ quan công an quận, huyện nơi cư trú để chụp ảnh, lấy vân tay.",
            "negs": [
                "Thủ tục cấp hộ chiếu phổ thông yêu cầu tờ khai điện tử, ảnh chân dung và căn cước công dân hợp lệ tại cơ quan xuất nhập cảnh.",
                "Quy trình đăng ký tạm trú cho người thuê nhà cần mang theo hợp đồng thuê nhà và bản sao căn cước công dân đến công an xã, phường.",
                "Đăng ký xe máy mới cần nộp lệ phí trước bạ và nộp hồ sơ bấm biển số tại công an xã hoặc quận huyện được phân cấp."
            ]
        },
        {
            "query": "Hồ Gươm nằm ở quận nào của Hà Nội?",
            "pos": "Hồ Gươm hay còn gọi là Hồ Hoàn Kiếm là một hồ nước ngọt tự nhiên nằm ở trung tâm quận Hoàn Kiếm, thủ đô Hà Nội.",
            "negs": [
                "Hồ Tây là hồ nước tự nhiên lớn nhất ở Hà Nội, nằm ở địa phận quận Tây Hồ với diện tích khoảng 500 ha.",
                "Công viên Thống Nhất là một trong những công viên lớn nhất Hà Nội, nằm tiếp giáp giữa các quận Đống Đa và Hai Bà Trưng.",
                "Chùa Một Cột là di tích lịch sử văn hóa nổi tiếng nằm gần Quảng trường Ba Đình tại quận Ba Đình, Hà Nội."
            ]
        },
        {
            "query": "Học máy học sâu và mạng nơ-ron khác nhau như thế nào?",
            "pos": "Học máy là nhánh rộng của trí tuệ nhân tạo, còn học sâu là tập con của học máy sử dụng các mạng nơ-ron sâu nhiều tầng để trích xuất đặc trưng tự động.",
            "negs": [
                "Kỹ thuật xử lý ngôn ngữ tự nhiên tập trung vào việc giúp máy tính hiểu và tương tác với ngôn ngữ loài người.",
                "Điện toán đám mây cung cấp tài nguyên tính toán như máy chủ ảo, lưu trữ dữ liệu thông qua mạng Internet.",
                "Thuật toán tìm kiếm theo chiều rộng duyệt đồ thị theo từng mức độ sâu trước khi đi xuống các nút con tiếp theo."
            ]
        },
        {
            "query": "Cách nấu phở bò Hà Nội chuẩn vị truyền thống?",
            "pos": "Nước dùng phở bò truyền thống được ninh kỹ từ xương ống bò cùng gừng nướng, hành nướng, quế, hoa hồi, thảo quả và gia vị vừa ăn trong nhiều giờ.",
            "negs": [
                "Cách làm bún chả Hà Nội gồm thịt ba chỉ ướp gia vị nướng than hoa, ăn kèm bún tươi, nước mắm chua ngọt và rau sống.",
                "Bánh mì pate trứng là món ăn sáng phổ biến nhanh gọn với vỏ bánh mì giòn tan, pate béo ngậy và dưa chuột muối chua.",
                "Món bún bò Huế đặc trưng bởi nước dùng cay nồng từ sả, ớt, mắm ruốc cùng sợi bún to và giò heo giòn dai."
            ]
        },
        {
            "query": "Thời gian ủ bệnh sốt xuất huyết dengue kéo dài bao lâu?",
            "pos": "Thời gian ủ bệnh sốt xuất huyết dengue thường kéo dài từ 4 đến 7 ngày sau khi bị muỗi mang virus chích, tối đa có thể lên tới 14 ngày.",
            "negs": [
                "Bệnh cảm cúm mùa thường lây lan qua giọt bắn đường hô hấp khi người bệnh ho hoặc hắt hơi gần người xung quanh.",
                "Để phòng ngừa dịch bệnh do muỗi truyền, người dân cần ngủ màn, diệt lăng quăng bọ gậy và đậy kín các dụng cụ chứa nước.",
                "Uống đủ nước và nghỉ ngơi hợp lý là biện pháp quan trọng hỗ trợ hạ sốt và bù dịch khi mắc các bệnh nhiễm siêu vi."
            ]
        }
    ]

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        while written < n_groups:
            tpl = templates[written % len(templates)]
            group = {
                "id": f"sample_{written+1:04d}",
                "query": norm_text(f"{tpl['query']} (mẫu {written+1})"),
                "documents": [
                    norm_text(tpl["pos"]),
                    norm_text(tpl["negs"][0]),
                    norm_text(tpl["negs"][1]),
                    norm_text(tpl["negs"][2]),
                ],
                "source": "sample_vietnamese"
            }
            f.write(json.dumps(group, ensure_ascii=False) + "\n")
            written += 1

    print(f"--> Đã tạo thành công {written} nhóm mẫu tại: {out_path}")


def stream_from_hf(out_path: str, n_groups: int = 20000, seed: int = 42):
    """
    Tải dữ liệu từ Hugging Face (streaming mode để không đầy RAM).
    Dataset: minhnguyent546/mmarco-vietnamese-split (triples/train)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Vui lòng cài đặt datasets bằng lệnh: pip install datasets")

    print(f"--> Đang kết nối streaming dataset mMARCO tiếng Việt (target: {n_groups} nhóm)...")
    ds = load_dataset(
        "minhnguyent546/mmarco-vietnamese-split",
        "triples",
        split="train",
        streaming=True,
    ).shuffle(seed=seed, buffer_size=10000)

    active = OrderedDict()
    completed = set()
    written = 0

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            q = norm_text(row.get("query", ""))
            p = norm_text(row.get("positive", ""))
            n = norm_text(row.get("negative", ""))

            # Lọc cơ bản
            if not (q and p and n) or n == p or len(q) < 6:
                continue

            key = query_positive_id(q, p)
            if key in completed:
                continue

            if key not in active:
                active[key] = {
                    "id": key,
                    "query": q,
                    "documents": [p],  # Vị trí 0 luôn là positive
                    "source": "mmarco_train"
                }

            group = active[key]
            if n not in group["documents"]:
                group["documents"].append(n)

            # Đủ 1 positive + 3 negatives = 4 documents
            if len(group["documents"]) == 4:
                f.write(json.dumps(group, ensure_ascii=False) + "\n")
                written += 1
                completed.add(key)
                del active[key]

                if written % 1000 == 0:
                    print(f"    Đã gom: {written}/{n_groups} nhóm...")

                if written >= n_groups:
                    break

            # Giữ kích thước buffer RAM không quá lớn
            if len(active) > 100000:
                active.popitem(last=False)

    print(f"--> Hoàn thành! Đã ghi {written} nhóm vào: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu nhóm K=4 cho Relational KD")
    parser.add_argument("--out", default="data/groups.jsonl", help="Đường dẫn file đầu ra")
    parser.add_argument("--n", type=int, default=20000, help="Số nhóm cần tạo")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--demo", action="store_true", help="Tạo nhanh file mẫu tiếng Việt để test")
    args = parser.parse_args()

    if args.demo:
        create_sample_data(args.out, n_groups=args.n if args.n < 1000 else 50)
    else:
        stream_from_hf(args.out, n_groups=args.n, seed=args.seed)


if __name__ == "__main__":
    main()
