import copy
import numpy as np
from typing import List, Dict, Tuple
from common.logger import Logger

logger = Logger.get_logger("merge_ocr_data")

# -----------------------------
# Box utilities
# -----------------------------
def bbox_to_xyxy(box: List[List[float]]) -> Tuple[float, float, float, float]:
    """
    Converts a 4-point polygon box to (x1, y1, x2, y2)
    """
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)

def coverage_ratio(small, large):
    sx1, sy1, sx2, sy2 = small
    lx1, ly1, lx2, ly2 = large

    inter_w = max(0, min(sx2, lx2) - max(sx1, lx1))
    inter_h = max(0, min(sy2, ly2) - max(sy1, ly1))
    inter = inter_w * inter_h

    small_area = max(0, sx2 - sx1) * max(0, sy2 - sy1)
    return inter / small_area if small_area > 0 else 0.0


def compute_iou(a: Tuple[float, float, float, float],
                b: Tuple[float, float, float, float]) -> float:
    """
    Computes IOU between two xyxy boxes
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def vertical_overlap_ratio(a, b) -> float:
    """
    Measures vertical alignment between two boxes
    """
    ay1, ay2 = a[1], a[3]
    by1, by2 = b[1], b[3]

    overlap = max(0, min(ay2, by2) - max(ay1, by1))
    min_h = min(ay2 - ay1, by2 - by1)

    return overlap / min_h if min_h > 0 else 0.0


def is_same_text(a, b, iou_thresh=0.3, y_overlap_thresh=0.6) -> bool:
    """
    Determines whether two boxes represent the same text
    """
    return (
        coverage_ratio(a, b) > iou_thresh or
        coverage_ratio(b, a) > iou_thresh
        # or vertical_overlap_ratio(a, b) >= y_overlap_thresh
    )


# -----------------------------
# JSON-safe numeric conversion
# -----------------------------
def to_python_number(x):
    if isinstance(x, np.generic):
        return x.item()
    return x


def merge_ocr_data(
    easy_ocr_data,
    tesseract_ocr_data,
    iou_threshold=0.1
):
    logger.info("Merging OCR data...")
    merged_data = copy.deepcopy(easy_ocr_data)
    tess_map = {item["uid"]: item for item in tesseract_ocr_data}

    for easy_item in merged_data:
        tess_item = tess_map.get(easy_item["uid"])
        if not tess_item:
            continue

        # Cache EasyOCR boxes
        easy_boxes = [
            bbox_to_xyxy(t["box"]) for t in easy_item["texts"]
        ]

        # 👉 iterate over a COPY
        for tess_text in tess_item["texts"][:]:
            tess_box = bbox_to_xyxy(tess_text["box"])

            overlap_found = False

            for idx, easy_box in enumerate(easy_boxes):
                if is_same_text(tess_box, easy_box, iou_threshold):
                    overlap_found = True

                    # print(
                    #     f"overlapping: tess_text: {tess_text['text']} "
                    #     f"==== easy_text: {easy_item['texts'][idx]['text']}"
                    # )

                    # # Optional: replace EasyOCR if Tesseract is more confident
                    # if tess_text.get("prob", 0) > easy_item["texts"][idx].get("prob", 0):
                    #     easy_item["texts"][idx]["text"] = tess_text["text"]
                    #     easy_item["texts"][idx]["prob"] = float(tess_text.get("prob", 0))

                    # ✅ REMOVE from Tesseract once matched
                    tess_item["texts"].remove(tess_text)
                    break

            # If NO overlap → append & also remove from Tesseract
            if not overlap_found:
                easy_item["texts"].append({
                    "box": tess_text["box"],
                    "text": tess_text["text"],
                    "prob": float(tess_text.get("prob", 0))
                })
                easy_boxes.append(tess_box)

                tess_item["texts"].remove(tess_text)

        # Stable reading order
        easy_item["texts"].sort(
            key=lambda t: (
                min(p[1] for p in t["box"]),
                min(p[0] for p in t["box"])
            )
        )

    return merged_data


def detect_and_merge_header_by_row_gap(texts, tolerance_ratio=0.35):
    logger.info("Detecting and merging header by row gap...")
    
    if len(texts) < 2:
        return texts, ""

    def box_to_xyxy(box):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        return min(xs), min(ys), max(xs), max(ys)

    # Compute gaps
    gaps = []
    for i in range(len(texts) - 1):
        _, _, _, y2 = box_to_xyxy(texts[i]["box"])
        _, y1, _, _ = box_to_xyxy(texts[i+1]["box"])
        gaps.append(max(0, y1 - y2))

    if len(gaps) < 2:
        return texts[1:], texts[0]["text"]

    # Estimate row gap (ignore first)
    min_row_gap = min(gaps[1:])
    first_gap = gaps[0]

    tolerance = min_row_gap * tolerance_ratio

    # Decide
    if first_gap > min_row_gap:
        # first line behaves like a row → single-line header
        return texts[1:], texts[0]["text"]

    # Multi-line header → merge first two
    t1, t2 = texts[0], texts[1]

    x1_1, y1_1, x2_1, y2_1 = box_to_xyxy(t1["box"])
    x1_2, y1_2, x2_2, y2_2 = box_to_xyxy(t2["box"])

    merged_box = [
        [min(x1_1, x1_2), min(y1_1, y1_2)],
        [max(x2_1, x2_2), min(y1_1, y1_2)],
        [max(x2_1, x2_2), max(y2_1, y2_2)],
        [min(x1_1, x1_2), max(y2_1, y2_2)]
    ]

    header_text = t1["text"] + " " + t2["text"]
    header_prob = max(t1["prob"], t2["prob"])

    return texts[2:], header_text