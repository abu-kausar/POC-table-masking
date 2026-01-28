import os
import cv2
import json
from data_extraction.data_extraction_tessaract import TessaractDataExtractor
from data_extraction.data_extractor_easyocr import EasyOcrDataExtractor

from utils.merge_ocr_data import merge_ocr_data
from utils.drawing import annotate_targeted_texts, draw_box_on_all_texts, mask_all_extracted_texts
from masking.masking_by_header import search_text_by_header
from masking.mask_all_match_text import search_by_matcher

if __name__ == "__main__":
    model_path = "assets/20260121_best.pt"
    easy_ocr_dex = EasyOcrDataExtractor(model_path)
    tessaract_ocr_dex = TessaractDataExtractor(model_path)
    
    img_path = "src/images/ss-1.jpeg"
    easy_ocr_data = easy_ocr_dex.data_extraction_from_image(img_path)
    tessaract_ocr_data = tessaract_ocr_dex.data_extraction_from_image(img_path)
    processed_data = merge_ocr_data(easy_ocr_data, tessaract_ocr_data, iou_threshold=0.3)
    
    # write processed data to further inspection
    os.makedirs("processed_data", exist_ok=True)
    with open("processed_data/processed_data.json", "w") as f:
        json.dump(processed_data, f, indent=4)
    
    # make outputs directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    annotated1 = draw_box_on_all_texts(img_path, processed_data, draw_bbox=True, fill_bbox_white=False)
    annotated2 = mask_all_extracted_texts(img_path, processed_data, draw_bbox=True, fill_bbox_white=True)
    
    
    # masking by header
    header_text = "Description"
    texts = search_text_by_header(processed_data, header_text)
    annotated3 = annotate_targeted_texts(img_path, texts, True, True)
    
    # write annotated images
    cv2.imwrite(os.path.join(output_dir, "annotated_all_texts.png"), cv2.cvtColor(annotated1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "masked_all_texts.png"), cv2.cvtColor(annotated2, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, f"masked_texts_by_header_{header_text}.png"), cv2.cvtColor(annotated3, cv2.COLOR_RGB2BGR))
    
    # # masking by matcher
    # texts = search_by_matcher(processed_data, "Touring Bike")
    # annotate_targeted_texts(img_path, texts, True, True)
    
    