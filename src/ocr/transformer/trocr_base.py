from src.ocr.base_ocr import TransformerOCR


class TrOCRBase(TransformerOCR):
    """TrOCR base-handwritten (Microsoft), via Hugging Face."""

    name = "trocr-base"
    checkpoint = "microsoft/trocr-base-handwritten"
