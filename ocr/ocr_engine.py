import io
import os
import re
import time

import easyocr
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageOps


DATA_PATH = os.environ.get("DATA_PATH", "./")
VIZAR_SERVER = os.environ.get("VIZAR_SERVER", "easyvizar.wings.cs.wisc.edu:5001")
OCR_LANGS = [x.strip() for x in os.environ.get("OCR_LANGS", "en").split(",") if x.strip()]
OCR_GPU = os.environ.get("OCR_GPU", "false").lower() == "true"
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.20"))


def encode_png(data):
    buffer = io.BytesIO()
    iio.imwrite(buffer, data, extension=".png")
    return buffer.getvalue()


def normalize_room_text(text):
    return re.sub(r"[^A-Za-z0-9]", "", text.upper())


def matches_room_number(text):
    patterns = [
        r"^[A-Z]?\d{2,4}[A-Z]?$",      # 221, B145, 314A
        r"^\d[A-Z]\d{2,3}$",           # 2W14
        r"^[A-Z]{1,4}\d{2,4}[A-Z]?$",  # RM221, CS314A
        r"^\d{2,4}$",                  # 221, 901
    ]
    return any(re.match(pattern, text) for pattern in patterns)


def looks_number_like(text):
    digit_count = sum(ch.isdigit() for ch in text)
    return digit_count >= 1 and len(text) <= 12


class OCRResult:
    def __init__(self, info, image, display_annotations=None):
        self.info = info
        self.image = image
        self.display_annotations = display_annotations or []

    def apply_masks(self):
        if self.image.ndim == 2:
            rgb = np.stack([self.image] * 3, axis=-1)
        else:
            rgb = self.image[:, :, :3]

        pil = Image.fromarray(rgb).convert("RGB")
        draw = ImageDraw.Draw(pil)

        h, w = rgb.shape[:2]

        for ann in self.display_annotations:
            boundary = ann["boundary"]

            left = int(boundary["left"] * w)
            top = int(boundary["top"] * h)
            right = int((boundary["left"] + boundary["width"]) * w)
            bottom = int((boundary["top"] + boundary["height"]) * h)

            label = ann.get("label", "text")
            sublabel = ann.get("sublabel", "")

            if label == "room-number":
                color = "red"
                text_y = max(0, top - 15)
            else:
                color = "yellow"
                text_y = min(h - 15, bottom + 2)

            display_text = sublabel if sublabel else label

            draw.rectangle([left, top, right, bottom], outline=color, width=3)
            draw.text((left, text_y), display_text, fill=color)

        annotated = np.array(pil)
        annotated_png = encode_png(annotated)

        mask = np.zeros((h, w, 4), dtype=np.uint8)
        mask_png = encode_png(mask)

        return annotated_png, mask_png


class OCREngine:
    def __init__(self):
        self.reader = None

    def initialize_model(self):
        if self.reader is None:
            self.reader = easyocr.Reader(OCR_LANGS, gpu=OCR_GPU)

    def choose_source(self, item):
        path = item.get("imagePath")
        url = item.get("imageUrl")

        if path not in [None, ""]:
            full_path = os.path.join(DATA_PATH, path)
            if os.path.isfile(full_path):
                return full_path

        if isinstance(url, str) and url.startswith("http"):
            return url

        if isinstance(url, str) and url.startswith("/"):
            return "http://" + VIZAR_SERVER + url

        raise Exception(f"Cannot load image path ({path}) or URL ({url})")

    def preprocess(self, image):
        if image.ndim == 2:
            return image

        rgb = image[:, :, :3]
        pil = Image.fromarray(rgb).convert("L")
        pil = ImageOps.autocontrast(pil)
        return np.array(pil)

    def _bbox_to_boundary(self, bbox, w, h):
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]

        min_x = max(0.0, min(xs))
        max_x = min(float(w), max(xs))
        min_y = max(0.0, min(ys))
        max_y = min(float(h), max(ys))

        if max_x <= min_x or max_y <= min_y:
            return None

        return {
            "left": float(min_x / w),
            "top": float(min_y / h),
            "width": float((max_x - min_x) / w),
            "height": float((max_y - min_y) / h),
        }

    def run(self, item):
        self.initialize_model()

        source = self.choose_source(item)
        print(f"Processing image from {source}...")

        image = iio.imread(source)
        h, w = image.shape[:2]

        preprocess_start = time.time()
        processed = self.preprocess(image)

        inference_start = time.time()
        raw_results = self.reader.readtext(processed)
        postprocess_start = time.time()

        print("Raw OCR results:")
        for entry in raw_results:
            print(entry)

        confirmed_annotations = []
        fallback_annotations = []

        for entry in raw_results:
            bbox, text, confidence = entry
            cleaned_text = normalize_room_text(text)

            print("OCR text:", text, "-> cleaned:", cleaned_text, "confidence:", confidence)

            boundary = self._bbox_to_boundary(bbox, w, h)
            if boundary is None:
                print("Rejected because bounding box is invalid")
                continue

            if confidence < MIN_CONFIDENCE:
                print("Rejected because confidence too low")
                continue

            if not cleaned_text:
                print("Rejected because cleaned text is empty")
                continue

            if matches_room_number(cleaned_text):
                annotation = {
                    "boundary": boundary,
                    "confidence": float(confidence),
                    "label": "room-number",
                    "sublabel": cleaned_text,
                }
                print("Accepted room-number annotation:", annotation)
                confirmed_annotations.append(annotation)
                continue

            if looks_number_like(cleaned_text):
                fallback = {
                    "boundary": boundary,
                    "confidence": float(confidence),
                    "label": "possible-room-text",
                    "sublabel": cleaned_text,
                }
                print("Stored fallback annotation:", fallback)
                fallback_annotations.append(fallback)
            else:
                print("Rejected because text does not match room-number pattern")

        postprocess_end = time.time()

        if len(raw_results) == 0:
            ocr_status = "no-text-detected"
            ocr_message = "Could not extract any text from image."
        elif len(confirmed_annotations) > 0:
            ocr_status = "room-number-detected"
            ocr_message = "Room number text extracted successfully."
        elif len(fallback_annotations) > 0:
            ocr_status = "possible-number-detected-no-room-match"
            ocr_message = "Detected number-like text regions, but could not confidently identify a room number."
        else:
            ocr_status = "text-detected-no-room-match"
            ocr_message = "Text was detected, but no room number could be confidently identified."

        print("OCR found", len(confirmed_annotations), "room-number annotations")
        print("OCR summary:", ocr_status, "-", ocr_message)

        if len(confirmed_annotations) > 0:
            annotations_for_server = confirmed_annotations
        elif len(fallback_annotations) > 0:
            annotations_for_server = fallback_annotations
        else:
            annotations_for_server = []

        info = {
            "status": "done",
            "annotations": annotations_for_server,
            "detector": {
                "model_repo": "ocr",
                "model_name": "easyocr",
                "engine_name": "easyocr",
                "preprocess_duration": inference_start - preprocess_start,
                "inference_duration": postprocess_start - inference_start,
                "postprocess_duration": postprocess_end - postprocess_start,
            },
            "ocr_summary": {
                "status": ocr_status,
                "message": ocr_message,
                "raw_text_count": len(raw_results),
                "matched_room_count": len(confirmed_annotations),
                "fallback_region_count": len(fallback_annotations),
                "raw_text": [entry[1] for entry in raw_results],
            },
        }

        return OCRResult(
            info=info,
            image=image,
            display_annotations=annotations_for_server,
        )