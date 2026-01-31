import os
import cv2
import json

from data_extraction.data_extraction_tessaract import TessaractDataExtractor
from data_extraction.data_extractor_easyocr import EasyOcrDataExtractor
from utils.merge_ocr_data import detect_and_merge_header_by_row_gap, merge_ocr_data
from utils.drawing import annotate_targeted_texts, draw_box_on_all_texts, mask_all_extracted_texts
from masking.masking_by_header import search_text_by_header
from masking.mask_all_match_text import search_by_matcher


def intermediate_drawing(img_path, processed_data, output_dir = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    """Draw intermediate annotated images for visualization."""
    annotated1 = draw_box_on_all_texts(img_path, processed_data, draw_bbox=True, fill_bbox_white=False)
    annotated2 = mask_all_extracted_texts(img_path, processed_data, draw_bbox=True, fill_bbox_white=True)
    # write annotated images
    cv2.imwrite(os.path.join(output_dir, "annotated_all_texts.png"), cv2.cvtColor(annotated1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "masked_all_texts.png"), cv2.cvtColor(annotated2, cv2.COLOR_RGB2BGR))


def masking_by_header(header_texts: list, processed_data, img_path, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    """Mask texts based on multiple headers."""
    # multiple headers may be provided, mask each one by one and show in single image
    texts_to_annotate = []
    for header_text in header_texts:
        texts = search_text_by_header(processed_data, header_text, match_threshold=0.7)
        texts_to_annotate.extend(texts)

    if not texts_to_annotate:
        print(f"Sorry! No texts found for headers: {header_texts}")
        return
    annotated = annotate_targeted_texts(img_path, texts_to_annotate, draw_bbox=True, fill_bbox_white=True)
    # save masking image
    cv2.imwrite(os.path.join(output_dir, f"masked_texts_by_headers.png"), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

def main(headers_text: list, img_path: str):
    model_path = "assets/best.pt"
    easy_ocr_dex = EasyOcrDataExtractor(model_path)
    tessaract_ocr_dex = TessaractDataExtractor(model_path)

    # make outputs directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Step-1: Extract data from both OCRs
    easy_ocr_data = easy_ocr_dex.data_extraction_from_image(img_path)
    tessaract_ocr_data = tessaract_ocr_dex.data_extraction_from_image(img_path)

    # Step-2: Merge Two OCR data
    processed_data = merge_ocr_data(easy_ocr_data, tessaract_ocr_data, iou_threshold=0.3)

    # processed_data = tessaract_ocr_data

    # Not mask header
    # Header selection
    for item in processed_data:
        if item["texts"]:
            item["texts"] = item["texts"][1:]

    # save processed data for further testing
    with open("outputs/processed_data.json", "w") as f:
        json.dump(processed_data, f, indent=4)

    # This drawing is optional, just for visualization
    intermediate_drawing(img_path, processed_data)

    # Step-4: Targeted masking
    # masking by single header
    masking_by_header(headers_text, processed_data, img_path, output_dir)

    # # masking by matcher
    # texts = search_by_matcher(processed_data, "Touring Bike")
    # annotate_targeted_texts(img_path, texts, True, True)




if __name__ == "__main__":

    headers_text = ["Batch Number"]
    image_path = "/content/drive/MyDrive/POC-table-masking/images/27.jpg"

    main(headers_text, image_path)