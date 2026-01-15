# Development Roadmap
## Phase 1 — MVP
- [ ] OpenCV + Tesseract
- [ ] Column-based masking
- [ ] REST API

## Phase 2 — Intelligence
- [ ] Table detection
- [ ] Partial masking
- [ ] User target parser

## Phase 3 — AI-first
- [ ] VLM target resolution
- [ ] Multi-language OCR
- [ ] Confidence feedback


# Development Plan
- [ ] Analysis image to find patterns
- [ ] Search best method to localize target labels or column name.
- [ ] Find field values based on localized target labels or column names.
- [ ] Mask field values.
- [ ] Build REST API around the core functionality.
- [ ] Improve accuracy with AI models.
- [ ] Add support for multiple languages.
- [ ] Optimize performance for large-scale document processing.

We will experiment the following table model:
| Model               | Purpose                          | Notes                             |
|---------------------|----------------------------------|-----------------------------------|
| OpenCV              | Image processing and table detection | Widely used, good community support |
| Tesseract OCR       | Text extraction from images      | Open-source, supports multiple languages |
| YOLOv5              | Object detection for table structure | Real-time detection capabilities   |
| LayoutLM            | Document layout understanding    | Pre-trained on document images     |
| PaddleOCR          | Alternative OCR engine           | High accuracy, easy to integrate   |
| Hugging Face Models | NLP tasks for target parsing     | Large model repository             |