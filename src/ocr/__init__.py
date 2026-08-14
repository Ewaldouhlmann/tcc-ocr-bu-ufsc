"""Módulo de modelos OCR: interface comum e implementações por categoria."""

from src.ocr.base_ocr import BaseOCR, ClassicalOCR, TransformerOCR, VLMOCR
from src.ocr.classical import DocTROCR, EasyOCR, PaddleOCR, TesseractOCR
from src.ocr.transformer import TrOCRBase, TrOCRLarge
from src.ocr.vlm import (
    DeepSeekOCR,
    GeminiOCR,
    GemmaOCR,
    LlamaParseFreeOCR,
    LlamaParsePremiumOCR,
    MistralOCR,
    QwenOCR,
)

#: Mapa nome -> classe, usado pelo benchmark para instanciar modelos a partir do
#: CLI (--models), com as mesmas chaves de benchmark/run_benchmark.py::AVAILABLE_MODELS.
MODEL_REGISTRY: dict[str, type[BaseOCR]] = {
    "tesseract": TesseractOCR,
    "easyocr": EasyOCR,
    "paddleocr": PaddleOCR,
    "doctr": DocTROCR,
    "trocr-base": TrOCRBase,
    "trocr-large": TrOCRLarge,
    "gemma4": GemmaOCR,
    "deepseek": DeepSeekOCR,
    "qwen": QwenOCR,
    "llamaparse-free": LlamaParseFreeOCR,
    "llamaparse-premium": LlamaParsePremiumOCR,
    "mistral": MistralOCR,
    "gemini": GeminiOCR,
}

__all__ = [
    "BaseOCR",
    "ClassicalOCR",
    "TransformerOCR",
    "VLMOCR",
    "TesseractOCR",
    "EasyOCR",
    "PaddleOCR",
    "DocTROCR",
    "TrOCRBase",
    "TrOCRLarge",
    "GemmaOCR",
    "DeepSeekOCR",
    "QwenOCR",
    "LlamaParseFreeOCR",
    "LlamaParsePremiumOCR",
    "MistralOCR",
    "GeminiOCR",
    "MODEL_REGISTRY",
]
