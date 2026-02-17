# -----------------------------
# OCR line merging (polygon boxes)
# -----------------------------

def box_stats(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]

    return {
        "x_min": min(xs),
        "x_max": max(xs),
        "y_min": min(ys),
        "y_max": max(ys),
        "height": max(ys) - min(ys),
    }


def vertical_overlap(box1, box2, overlap_ratio=0.5):
    b1 = box_stats(box1)
    b2 = box_stats(box2)

    overlap = max(
        0,
        min(b1["y_max"], b2["y_max"]) - max(b1["y_min"], b2["y_min"])
    )

    return overlap >= overlap_ratio * min(b1["height"], b2["height"])


def merge_boxes(boxes):
    xs, ys = [], []
    for box in boxes:
        for x, y in box:
            xs.append(x)
            ys.append(y)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    return [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
    ]


def merge_same_line_text(ocr_items, overlap_ratio=0.5):
    # Sort roughly top → bottom
    ocr_items.sort(key=lambda x: box_stats(x["box"])["y_min"])

    lines = []

    # Group into lines
    for item in ocr_items:
        placed = False
        for line in lines:
            if vertical_overlap(item["box"], line[0]["box"], overlap_ratio):
                line.append(item)
                placed = True
                break

        if not placed:
            lines.append([item])

    merged_output = []

    # Merge each line
    for line in lines:
        # Left → right order
        line.sort(key=lambda x: box_stats(x["box"])["x_min"])

        merged_text = " ".join(w["text"] for w in line)
        merged_box = merge_boxes([w["box"] for w in line])
        merged_conf = min(w["prob"] for w in line)

        merged_output.append({
            "text": merged_text,
            "box": merged_box,
            "prob": merged_conf
        })

    return merged_output

def merge_texts(processed_data):
    """Merge horizontally aligned texts into line-level boxes."""
    for item in processed_data:
        if item["texts"]:
            if item["type"] == "top_down_text_field":
                item["texts"] = merge_same_line_text(item["texts"])
            # elif item["type"] == "text_field":
            #     item["texts"] = merge_same_line_text(item["texts"])
            # elif item["type"] == "text_info":
            #     item["texts"] = merge_same_line_text(item["texts"])
            # elif item["type"] == "table_column":
            #     item["texts"] = merge_same_line_text(item["texts"])

    return processed_data
