# extract all boxes related to user target text

# text to be masked
# targeted_text = "Touring Bike"
# extract all boxes related to user target text

# text to be masked
# targeted_text = "Touring Bike"

import re
from rapidfuzz import fuzz
from collections import Counter

# ------------------------------------------------
# 1. Normalize OCR text
# ------------------------------------------------
def normalize_text(text: str) -> str:
    ocr_map = {
        '0': 'o',
        '1': 'l',
        '5': 's',
        '8': 'b',
        'h': 'n',
    }

    text = text.lower()

    # Replace common OCR mistakes
    for k, v in ocr_map.items():
        text = text.replace(k, v)

    # Remove special characters
    text = re.sub(r'[^a-z\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ------------------------------------------------
# 2. Token-based similarity
# ------------------------------------------------
def token_similarity(a: str, b: str) -> float:
    return fuzz.token_set_ratio(a, b) / 100.0


# ------------------------------------------------
# 3. Edit distance similarity
# ------------------------------------------------
def edit_similarity(a: str, b: str) -> float:
    return fuzz.ratio(a, b) / 100.0


# ------------------------------------------------
# 4. Character N-gram similarity (Jaccard)
# ------------------------------------------------
def ngram_similarity(a: str, b: str, n: int = 3) -> float:
    def ngrams(text):
        return [text[i:i+n] for i in range(len(text) - n + 1)]

    a_ngrams = Counter(ngrams(a))
    b_ngrams = Counter(ngrams(b))

    intersection = sum((a_ngrams & b_ngrams).values())
    union = sum((a_ngrams | b_ngrams).values())

    return intersection / union if union != 0 else 0.0


# ------------------------------------------------
# 5. Final matcher
# ------------------------------------------------
def ocr_text_match(expected_text: str, ocr_text: str, threshold: float = 0.5):
    norm_expected = normalize_text(expected_text)
    norm_ocr = normalize_text(ocr_text)

    token_score = token_similarity(norm_expected, norm_ocr)
    edit_score = edit_similarity(norm_expected, norm_ocr)
    ngram_score = ngram_similarity(norm_expected, norm_ocr)

    final_score = (
        0.4 * token_score +
        0.3 * edit_score +
        0.3 * ngram_score
    )

    return {
        "normalized_expected": norm_expected,
        "normalized_ocr": norm_ocr,
        "token_score": round(token_score, 3),
        "edit_score": round(edit_score, 3),
        "ngram_score": round(ngram_score, 3),
        "final_score": round(final_score, 3),
        "match": final_score >= threshold
    }

# Find best match
def find_best_match(target_text, ocr_text_list, threshold=0.5):
    best_result = None
    best_score = 0.0

    for ocr_text in ocr_text_list:
        result = ocr_text_match(target_text, ocr_text, threshold)
        if result["final_score"] > best_score:
            best_score = result["final_score"]
            best_result = {
                "ocr_text": ocr_text,
                **result
            }

    if best_result and best_score >= threshold:
        return best_result

    return None


def search_text_by_header(
    processed_data,
    header_name,
    match_threshold=0.7
):
    print("="*20)
    print(f"Searching texts under header: '{header_name}' with match threshold: {match_threshold}\n")
    if processed_data is None or not processed_data:
        print("No processed data available.\n")
        return []
    # first attempt
    texts = searching_attemp(
        processed_data,
        header_name,
        match_threshold
    )
    # second attemp if not found
    if texts:
        return texts
    else:
        # take first item from processed_data and concanate it search term
        if processed_data[0]["texts"]:
            print(f"Trying with combined header..........\n")
            texts = searching_attemp(
                processed_data,
                header_name,
                match_threshold,
                is_combined=True
            )
    # make third attempt if still not found
    if texts:
        # since found in second attempt
        # delete first element texts to avoid duplicate masking
        texts = texts[1:]
        return texts
    else:
        print("Second attempt failed, trying with decreasing match threshold...\n")
        print("="*20)
        # Decrease the match threshold and try again
        for new_threshold in [0.65, 0.60, 0.55, 0.50]:
            print(f"Trying with match threshold: {new_threshold}\n")
            
            texts = searching_attemp(
                processed_data,
                header_name,
                match_threshold=new_threshold
            )
            if texts:
                return texts
    if not texts:
        print("Sorry can't find the expected header!")
        return []
    
    return texts

def searching_attemp(
    processed_data,
    header_name,
    match_threshold=0.7,
    is_combined=False
) -> list:
    texts = []
    normalized_target = normalize_text(header_name)

    for item in processed_data:
        raw_header = item.get("header", "").strip()
        first_text = item['texts'][0]['text'] if item['texts'] else ""
        if is_combined:
            raw_header = f"{raw_header} {first_text}" if item["texts"] else raw_header

        if not raw_header:
            continue

        # --------------------------------------------------
        # 🔍 OCR-AWARE HEADER MATCHING
        # --------------------------------------------------
        match_result = ocr_text_match(
            expected_text=header_name,
            ocr_text=raw_header,
            threshold=match_threshold
        )

        if not match_result["match"]:
            continue  # ❌ Not the target header

        # --------------------------------------------------
        # ✅ HEADER MATCHED — extract child texts
        # --------------------------------------------------
        parent_x1, parent_y1, _, _ = item["box"]

        # if combined match, skip first text as it is part of header
        flag = True if is_combined else False
        
        for text_info in item["texts"]:
            if "text" not in text_info or "box" not in text_info:
                continue

            text_content = text_info["text"]
    
            # Skip if this text is basically the header itself
            header_match = ocr_text_match(
                expected_text=raw_header,
                ocr_text=text_content,
                threshold=match_threshold
            )
            if header_match["match"]:
                continue

            # EasyOCR bbox format: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            points = text_info["box"]

            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]

            # Convert to absolute image coordinates
            x1_abs = int(parent_x1 + min(x_coords))
            y1_abs = int(parent_y1 + min(y_coords))
            x2_abs = int(parent_x1 + max(x_coords))
            y2_abs = int(parent_y1 + max(y_coords))

            texts.append(
                (
                    text_content,
                    [[x1_abs, y1_abs], [x2_abs, y2_abs]]
                )
            )

    return texts

