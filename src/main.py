import cv2
import easyocr
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
import os

# Set up logging for production-grade tracking
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def merge_ocr_results(ocr_results: List[Tuple], distance_threshold: float = 20) -> List[Tuple]:
    """
    Merge OCR bounding boxes that overlap or are close to each other.
    """
    try:
        if not ocr_results:
            return []

        # Convert bboxes to rectangles for easier processing
        boxes = []
        for bbox, text, conf in ocr_results:
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            boxes.append({
                'x_min': x_min, 'x_max': x_max,
                'y_min': y_min, 'y_max': y_max,
                'text': text, 'conf': conf,
                'merged': False
            })

        # Sort boxes by y position (top to bottom), then x position (left to right)
        boxes.sort(key=lambda b: (b['y_min'], b['x_min']))

        merged_results = []
        for i, box in enumerate(boxes):
            if box['merged']:
                continue

            merged_group = [box]
            box['merged'] = True

            for j in range(i + 1, len(boxes)):
                other = boxes[j]
                if other['merged']:
                    continue

                group_x_min = min(b['x_min'] for b in merged_group)
                group_x_max = max(b['x_max'] for b in merged_group)
                group_y_min = min(b['y_min'] for b in merged_group)
                group_y_max = max(b['y_max'] for b in merged_group)

                horizontal_gap = max(other['x_min'], group_x_min) - min(other['x_max'], group_x_max)
                horizontal_overlap = horizontal_gap < distance_threshold

                vertical_gap = max(other['y_min'], group_y_min) - min(other['y_max'], group_y_max)
                vertical_overlap = vertical_gap < distance_threshold

                y_center_group = (group_y_min + group_y_max) / 2
                y_center_other = (other['y_min'] + other['y_max']) / 2
                same_line = abs(y_center_group - y_center_other) < distance_threshold

                if (horizontal_overlap and vertical_overlap) or (same_line and horizontal_overlap):
                    merged_group.append(other)
                    other['merged'] = True

            merged_x_min = min(b['x_min'] for b in merged_group)
            merged_x_max = max(b['x_max'] for b in merged_group)
            merged_y_min = min(b['y_min'] for b in merged_group)
            merged_y_max = max(b['y_max'] for b in merged_group)

            merged_group.sort(key=lambda b: (b['y_min'], b['x_min']))
            merged_text = ' '.join(b['text'] for b in merged_group)
            merged_conf = sum(b['conf'] for b in merged_group) / len(merged_group)

            merged_bbox = [
                [merged_x_min, merged_y_min],
                [merged_x_max, merged_y_min],
                [merged_x_max, merged_y_max],
                [merged_x_min, merged_y_max]
            ]
            merged_results.append((merged_bbox, merged_text, merged_conf))

        logger.info(f"OCR Merging: {len(ocr_results)} boxes -> {len(merged_results)} boxes")
        return merged_results

    except Exception as e:
        logger.error(f"Error in merge_ocr_results: {str(e)}")
        return ocr_results


def clip_and_split_ocr_boxes(merged_ocr: List[Tuple], columns: List[Dict]) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Clip and split OCR bounding boxes to prevent overflow across column boundaries.
    """
    clipped_results = []
    outside_results = []
    
    try:
        if not columns:
            return [], [box for box in merged_ocr]

        for (bbox, text, conf) in merged_ocr:
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            bb_x_min, bb_x_max = min(x_coords), max(x_coords)
            bb_y_min, bb_y_max = min(y_coords), max(y_coords)

            overlapping_columns = []
            for col in columns:
                x_overlap = not (bb_x_max < col['x_start'] or bb_x_min > col['x_end'])
                y_overlap = not (bb_y_max < col['y_start'] or bb_y_min > col['y_end'])

                if x_overlap and y_overlap:
                    overlap_x_min = max(bb_x_min, col['x_start'])
                    overlap_x_max = min(bb_x_max, col['x_end'])
                    overlap_y_min = max(bb_y_min, col['y_start'])
                    overlap_y_max = min(bb_y_max, col['y_end'])
                    overlap_area = (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)

                    overlapping_columns.append({
                        'col': col, 'overlap_area': overlap_area,
                        'x_min': overlap_x_min, 'x_max': overlap_x_max,
                        'y_min': overlap_y_min, 'y_max': overlap_y_max
                    })

            if not overlapping_columns:
                outside_results.append((bbox, text, conf))
                continue

            # Sort by overlap area to assign primary text
            overlapping_columns.sort(key=lambda x: x['overlap_area'], reverse=True)

            for idx, overlap in enumerate(overlapping_columns):
                clipped_bbox = [
                    [overlap['x_min'], overlap['y_min']],
                    [overlap['x_max'], overlap['y_min']],
                    [overlap['x_max'], overlap['y_max']],
                    [overlap['x_min'], overlap['y_max']]
                ]
                split_text = text if idx == 0 else ""
                clipped_results.append((clipped_bbox, split_text, conf, overlap['col']['id']))

        return clipped_results, outside_results

    except Exception as e:
        logger.error(f"Error in clip_and_split_ocr_boxes: {str(e)}")
        return [], []

def draw_all_columns(img: np.ndarray, columns: List[Dict], box_width: int = 3) -> np.ndarray:
    try:
        if img is None:
            raise ValueError("Input image is None")
        
        result = img.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

        for idx, col in enumerate(columns):
            color = colors[idx % len(colors)]
            cv2.rectangle(result, (col['x_start'], col['y_start']), (col['x_end'], col['y_end']), color, box_width)
            cv2.putText(result, f"Col {col['id']}", (col['x_start'] + 5, col['y_start'] + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return result
    except Exception as e:
        logger.error(f"Error in draw_all_columns: {str(e)}")
        return img

def draw_clipped_ocr_boxes(img: np.ndarray, clipped_ocr: List[Tuple],
                           columns: List[Dict], box_width: int = 2) -> np.ndarray:
    """
    Draw all clipped OCR bounding boxes with column-matched colors.

    Args:
        img: Input image
        clipped_ocr: List of (bbox, text, confidence, column_id) tuples
        columns: List of column dictionaries
        box_width: Width of bounding box lines

    Returns:
        Image with OCR boxes drawn
    """
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 128, 0),  # Orange
        (0, 128, 255),  # Light Blue
        (128, 255, 0),  # Lime
    ]

    result = img.copy()

    for (bbox, text, conf, col_id) in clipped_ocr:
        color = colors[col_id % len(colors)]
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(result, [pts], isClosed=True, color=color, thickness=box_width)

        # Optionally draw text label
        if text:
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            text_pos = (int(min(x_coords)) + 2, int(min(y_coords)) + 12)
            cv2.putText(result, text[:20], text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, color, 1)

    print(f"🎨 Drew {len(clipped_ocr)} clipped OCR boxes")
    return result


def draw_outside_ocr_boxes(img: np.ndarray, outside_ocr: List[Tuple],
                          box_width: int = 3, color: Tuple = (0, 0, 255)) -> np.ndarray:
    """
    Draw OCR bounding boxes that are outside all columns.

    Args:
        img: Input image
        outside_ocr: List of (bbox, text, confidence) tuples for outside text
        box_width: Width of bounding box lines
        color: Color for outside boxes (default: red)

    Returns:
        Image with outside OCR boxes drawn
    """
    result = img.copy()

    for (bbox, text, conf) in outside_ocr:
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(result, [pts], isClosed=True, color=color, thickness=box_width)

    print(f"🎨 Drew {len(outside_ocr)} outside OCR boxes in red")
    return result

def normalize_for_match(text: str) -> str:
    if not text: return ""
    # Remove common OCR noise and force lowercase
    return ''.join(c for c in text.lower() if c.isalnum())

def mask_column_by_text(filename: str, search_terms: List[str], outside_search_terms: List[str] = None, **kwargs):
    """
    Main orchestration function with global exception handling.
    """
    # 1. Extract parameters from kwargs with safe defaults
    h_kernel_size = kwargs.get('horizontal_kernel_size', 40)
    v_kernel_size = kwargs.get('vertical_kernel_size', 40)
    language = kwargs.get('language', ['en'])
    gpu = kwargs.get('gpu', False)
    merge_distance = kwargs.get('merge_distance', 20)
    outside_search_terms = outside_search_terms or []

    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Image not found: {filename}")

        img = cv2.imread(filename)
        if img is None:
            raise ValueError(f"OpenCV could not decode image: {filename}")
        
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Binary thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)

        # === STEP 1: Detect table structure ===
        # Use the extracted kernel sizes
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_size, 1))
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)

        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_size))
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

        # ... (Internal functions extract_segments and merge_segments remain the same)
        def extract_segments(binary_img, is_horizontal):
            contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            segments = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if is_horizontal:
                    segments.append({'start': x, 'end': x + w, 'position': y + h // 2})
                else:
                    segments.append({'start': y, 'end': y + h, 'position': x + w // 2})
            return segments

        def merge_segments(segments, pos_tol, gap_tol, img_size):
            if not segments: return []
            segments = sorted(segments, key=lambda s: (s['position'], s['start']))
            merged = []
            for seg in segments:
                merged_flag = False
                for existing in merged:
                    pos_diff = abs(seg['position'] - existing['position'])
                    if pos_diff <= img_size * pos_tol:
                        gap = max(seg['start'], existing['start']) - min(seg['end'], existing['end'])
                        if gap <= img_size * gap_tol:
                            existing['start'] = min(existing['start'], seg['start'])
                            existing['end'] = max(existing['end'], seg['end'])
                            merged_flag = True
                            break
                if not merged_flag:
                    merged.append(seg.copy())
            return merged

        h_segs = extract_segments(h_lines, True)
        v_segs = extract_segments(v_lines, False)
        h_merged = merge_segments(h_segs, 0.02, 0.10, height)
        v_merged = merge_segments(v_segs, 0.02, 0.10, width)

        # Intersection logic...
        intersections = []
        for h in h_merged:
            for v in v_merged:
                if h['start'] <= v['position'] <= h['end'] and v['start'] <= h['position'] <= v['end']:
                    intersections.append((v['position'], h['position']))

        if not intersections:
            logger.warning("No table structure detected")
            return {'all_columns': img, 'final_result': img}

        x_vals = sorted(list(set([pt[0] for pt in intersections])))
        y_vals = [pt[1] for pt in intersections]
        y_min, y_max = min(y_vals), max(y_vals)

        merged_x = []
        for x in x_vals:
            if not merged_x or abs(x - merged_x[-1]) > width * 0.02:
                merged_x.append(x)
            else:
                merged_x[-1] = (merged_x[-1] + x) // 2

        columns = []
        for i in range(len(merged_x) - 1):
            columns.append({
                'id': i, 'x_start': merged_x[i], 'x_end': merged_x[i + 1],
                'y_start': y_min, 'y_end': y_max
            })

        # === STEP 2: OCR ===
        reader = easyocr.Reader(language, gpu=gpu)
        ocr_results = reader.readtext(filename)
        merged_ocr = merge_ocr_results(ocr_results, merge_distance)
        clipped_ocr, outside_ocr = clip_and_split_ocr_boxes(merged_ocr, columns)

        # === STEP 3: Separate headers from data ===
        # Store OCR results per column
        ocr_by_column = {i: [] for i in range(len(columns))}

        for (bbox, text, conf, col_id) in clipped_ocr:
            ocr_by_column[col_id].append((bbox, text, conf))

        headers_by_column = {}
        data_by_column = {}

        for col_id, ocr_boxes in ocr_by_column.items():
            if not ocr_boxes:
                continue

            # Sort by y position to find topmost box (header)
            sorted_boxes = sorted(ocr_boxes, key=lambda x: min(p[1] for p in x[0]))

            # First box is the header
            headers_by_column[col_id] = sorted_boxes[0]

            # Rest are data
            data_by_column[col_id] = sorted_boxes[1:] if len(sorted_boxes) > 1 else []

            if sorted_boxes:
                header_text = sorted_boxes[0][1]
                print(f"Column {col_id} header: '{header_text}' with {len(sorted_boxes)-1} data rows")

        # === STEP 4: Search for EXACT matches in HEADERS only ===
        # search_normalized = [term.strip().lower() for term in search_terms]
        search_normalized = [normalize_for_match(term) for term in search_terms]
        matched_cols = []
        matched_ids = set()

        # Track which terms matched which columns
        matches_by_term = {term: [] for term in search_terms}

        # Search in HEADERS with EXACT match
        print("\n🔍 Searching for EXACT matches in column headers...")

        for col_id, (bbox, text, conf) in headers_by_column.items():
            header_norm = normalize_for_match(text)

            for original_term, term_norm in zip(search_terms, search_normalized):
                # Match if the normalized header equals the normalized term 
                # OR if one is contained within the other
                if term_norm == header_norm or term_norm in header_norm or header_norm in term_norm:
                    col = next((c for c in columns if c['id'] == col_id), None)
                    if col and col['id'] not in matched_ids:
                        matched_cols.append(col)
                        matched_ids.add(col['id'])
                        matches_by_term[original_term].append(('column', col_id, text))
                        print(f"✅ MATCH FOUND: '{text}' matched with '{original_term}'")

        # === STEP 5: Search for EXACT matches in OUTSIDE text AND HEADERS ===
        outside_search_normalized = [normalize_for_match(term) for term in outside_search_terms]
        matched_outside = []

        # Track which outside terms matched
        outside_matches_by_term = {term: [] for term in outside_search_terms}

        print("\n🔍 Searching for matches in outside text AND headers...")

        # Search in OUTSIDE text boxes
        for (bbox, text, conf) in outside_ocr:
            # Normalize the text found by OCR
            text_norm = normalize_for_match(text)

            for original_term, term_norm in zip(outside_search_terms, outside_search_normalized):
                # Check for exact match OR containment (handles "Branch/Plant:" with colon)
                if term_norm != "" and (term_norm == text_norm or term_norm in text_norm or text_norm in term_norm):
                    matched_outside.append((bbox, text, conf))
                    outside_matches_by_term[original_term].append(('outside', None, text))
                    print(f"✅ MATCH: Found '{text}' (matching '{original_term}') OUTSIDE columns")
                    break

        # Log if search terms appear in headers (for debugging) but DON'T mask them
        for col_id, (bbox, text, conf) in headers_by_column.items():
            header_norm = normalize_for_match(text)

            for original_term, term_norm in zip(search_terms, search_normalized):
                # Match if the normalized header equals the normalized term 
                # OR if one is contained within the other
                if term_norm == header_norm or term_norm in header_norm or header_norm in term_norm:
                    col = next((c for c in columns if c['id'] == col_id), None)
                    if col and col['id'] not in matched_ids:
                        matched_cols.append(col)
                        matched_ids.add(col['id'])
                        matches_by_term[original_term].append(('column', col_id, text))
                        print(f"✅ MATCH FOUND: '{text}' matched with '{original_term}'")
                        break

        # === STEP 6: Draw visualization - Cover DATA ONLY with white and *** ===
        result = img.copy()

        # Cover data boxes ONLY in matched columns with white and ***
        # Headers are NOT masked
        for col_id in matched_ids:
            data_boxes = data_by_column.get(col_id, [])
            for (bbox, text, conf) in data_boxes:
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                x_min, y_min = int(min(x_coords)), int(min(y_coords))
                x_max, y_max = int(max(x_coords)), int(max(y_coords))

                # Draw filled white rectangle to cover the bounding box
                cv2.rectangle(result, (x_min, y_min), (x_max, y_max), (255, 255, 255), -1)

                # Draw *** left-aligned with some padding
                padding = 5
                text_y = y_min + (y_max - y_min + 12) // 2  # Vertically centered
                cv2.putText(result, "***", (x_min + padding, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        if matched_ids:
            print(f"\n🔒 Covered DATA boxes in {len(matched_ids)} matched column(s) with white and ***")
            print(f"   Headers were NOT masked")

        # Cover matched outside boxes and extend 300px to the right
        if matched_outside:
            outside_color = (255, 255, 255)  # White for covering
            for (bbox, text, conf) in matched_outside:
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                x_min, y_min = int(min(x_coords)), int(min(y_coords))
                x_max, y_max = int(max(x_coords)), int(max(y_coords))

                # Extend 300px to the right from the end of the box (but don't exceed image width)
                extended_x_max = min(x_max + 300, width)

                # Draw filled white rectangle covering the bounding box and extension
                cv2.rectangle(result, (x_max, y_min), (extended_x_max, y_max),
                            outside_color, -1)

                # Draw *** left-aligned in the original box area with padding
                padding = 5
                text_y = y_min + (y_max - y_min + 12) // 2  # Vertically centered
                cv2.putText(result, "***", (x_max + padding + 50, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                print(f"🔒 Covered outside box '{text}' and extended 300px right")

            print(f"✅ Processed {len(matched_outside)} matched outside box(es)")

        # Print summary
        print(f"\n{'='*60}")
        print("SEARCH SUMMARY")
        print(f"{'='*60}")

        # Column header search results
        if search_terms:
            print("\n📋 COLUMN HEADER SEARCH:")
            for term, matches in matches_by_term.items():
                if matches:
                    col_matches = [m[1] for m in matches if m[0] == 'column']
                    match_str = f"  '{term}':"
                    if col_matches:
                        match_str += f" ✅ Found in column header(s) {sorted(set(col_matches))}"
                    print(match_str)
                else:
                    print(f"  '{term}': ❌ Not found (exact match required)")

        # Outside text search results
        if outside_search_terms:
            print("\n🔍 OUTSIDE TEXT SEARCH:")
            for term, matches in outside_matches_by_term.items():
                if matches:
                    print(f"  '{term}': ✅ Found OUTSIDE columns ({len(matches)} match(es))")
                else:
                    print(f"  '{term}': ❌ Not found (exact match required)")

        if not matched_cols and not matched_outside:
            print(f"\n❌ No EXACT matches found for any search terms")
            print(f"   Note: Search is case-insensitive but requires exact text match after lowering/stripping")

        return {
            'all_columns': draw_all_columns(img, columns),
            'final_result': result # This is the image with white boxes and ***
        }

    except Exception as e:
        logger.error(f"FATAL: mask_column_by_text failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc()) # Helps debug exactly where it fails
        return {'final_result': img}


# Example usage
if __name__ == "__main__":
    # Input image path
    input_image = 'src/images/ss-2.jpeg'
    
    results = mask_column_by_text(
        filename=input_image,
        search_terms=['line number', 'description',],
        outside_search_terms=['order number', 'sold to', 'ship to', 'branch/plant'],
        box_width=5,
        merge_distance=5,
        overlay_alpha=0.6
    )

    # Get the directory and base name of the input image
    import os
    input_dir = os.path.dirname(input_image)
    base_name = os.path.splitext(os.path.basename(input_image))[0]
    
    # Save results in the same directory as input image
    output_paths = {
        'final_result': os.path.join(input_dir, f'{base_name}_final_result.jpg')
    }
    
    cv2.imwrite(output_paths['final_result'], results['final_result'])
