# load train yolo model and predict on image
import os
import cv2
import numpy as np
from detection.ui_detector import UIElementDetector
import pytesseract

class TessaractDataExtractor:
    def __init__(self, model_path: str):
        self.tesseract_config = "--oem 3 --psm 12"
        self.ui_element_extractor = UIElementDetector(model_path)

    # -----------------------------
    # UI detection
    # -----------------------------
    def get_data_from_image(self, image_path: str):
        extracted_data = self.ui_element_extractor.predict(image_path)
        processed_data = []

        for i, item in enumerate(extracted_data):
            processed_data.append({
                "uid": i + 1,
                "box": item["box"],        # [x1, y1, x2, y2]
                "texts": [],
                "header": "",
                "type": item["label"]
            })

        return processed_data

    # -----------------------------
    # Main OCR pipeline
    # -----------------------------
    def data_extraction_from_image(self, img_path: str):
        processed_data = self.get_data_from_image(img_path)
        original_image = cv2.imread(img_path)

        if original_image is None:
            print(f"Error: Could not load image from {img_path}")
            return processed_data

        print("Performing Tesseract OCR on detected UI elements...\n")

        for item in processed_data:
            # adding padding 5 and also ensure the padding value does not go beyond image boundaries
            x1, y1, x2, y2 = map(int, item["box"])
            cropped_x1 = max(0, x1)
            cropped_y1 = max(0, y1)
            cropped_x2 = min(original_image.shape[1], x2)
            cropped_y2 = min(original_image.shape[0], y2)
            cropped_image = original_image[cropped_y1:cropped_y2, cropped_x1:cropped_x2]

            if cropped_image.size == 0:
                item["texts"] = []
                continue

            data = pytesseract.image_to_data(
                cropped_image,
                lang="eng",
                config=self.tesseract_config,  # e.g. "--oem 3 --psm 12"
                output_type=pytesseract.Output.DICT
            )

            lines = {}

            # -----------------------------
            # Group words into lines
            # -----------------------------
            for i in range(len(data["text"])):
                text = data["text"][i].strip()
                conf = int(data["conf"][i])

                # level 5 = word
                if not text or conf <= 0.1 or data["level"][i] != 5:
                    continue

                key = (
                    data["block_num"][i],
                    data["par_num"][i],
                    data["line_num"][i]
                )

                lines.setdefault(key, []).append(i)

            line_texts = []

            # -----------------------------
            # Build line-level text + box
            # -----------------------------
            for indices in lines.values():
                texts = [data["text"][i] for i in indices]
                text = " ".join(texts)
                # print(text)
                x_min = min(data["left"][i] for i in indices)
                y_min = min(data["top"][i] for i in indices)
                x_max = max(data["left"][i] + data["width"][i] for i in indices)
                y_max = max(data["top"][i] + data["height"][i] for i in indices)

                line_texts.append({
                    "text": text,
                    "box": [
                        [x_min, y_min],
                        [x_max, y_min],
                        [x_max, y_max],
                        [x_min, y_max],
                    ],
                    "prob": min(int(data["conf"][i]) for i in indices) / 100.0
                })

            # -----------------------------
            # Sort lines top → bottom
            # -----------------------------
            line_texts.sort(key=lambda x: x["box"][0][1])

            item["texts"] = line_texts

        return processed_data
    
    @staticmethod
    def texts_extraction_from_image(img_path: str):
        original_image = cv2.imread(img_path)

        if original_image is None:
            print(f"Error: Could not load image from {img_path}")
            return []

        data = pytesseract.image_to_data(
            original_image,
            lang="eng",
            config="--oem 3 --psm 12",  # e.g. "--oem 3 --psm 12"
            output_type=pytesseract.Output.DICT
        )

        processed_data = []

        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            conf = float(data["conf"][i])

            # level 5 = word
            if not text or conf <= 0 or data["level"][i] != 5:
                continue

            x_min = data["left"][i]
            y_min = data["top"][i]
            x_max = x_min + data["width"][i]
            y_max = y_min + data["height"][i]

            box = [
                [x_min, y_min],  # top-left
                [x_max, y_min],  # top-right
                [x_max, y_max],  # bottom-right
                [x_min, y_max]   # bottom-left
            ]

            processed_data.append([text, box])

        return processed_data

