"""Módulo de modelos OCR: interface comum e implementações por categoria."""

from src.ocr.base_ocr import BaseOCR, ClassicalOCR, TransformerOCR, VLMOCR
from src.ocr.classical import DocTROCR, EasyOCR, PaddleOCR, TesseractOCR
from src.ocr.transformer import TrOCRBase, TrOCRLarge
from src.ocr.vlm.nuvem import DeepSeekOCRModel, Gemma4E4BOCR, PaddleOCRVLOCR, Qwen7BOCR
from src.ocr.vlm import (
    DeepSeekOCR,
    GeminiOCR,
    GemmaOCR,
    GLMOCR,
    LlamaParseFreeOCR,
    LlamaParseLayoutAgentOCR,
    LlamaParseLVMOCR,
    LlamaParsePremiumOCR,
    MistralOCR,
    Qwen3OCR,
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
    "qwen3-vl": Qwen3OCR,
    "glm-ocr": GLMOCR,
    # Exigem GPU: executados na nuvem (notebooks/benchmark_gpu.ipynb)
    "qwen-7b": Qwen7BOCR,
    "gemma4-e4b": Gemma4E4BOCR,
    "deepseek-ocr": DeepSeekOCRModel,
    "paddleocr-vl": PaddleOCRVLOCR,
    "llamaparse-free": LlamaParseFreeOCR,
    "llamaparse-premium": LlamaParsePremiumOCR,
    "llamaparse-lvm": LlamaParseLVMOCR,
    "llamaparse-layout": LlamaParseLayoutAgentOCR,
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
    "Qwen3OCR",
    "GLMOCR",
    "LlamaParseFreeOCR",
    "LlamaParsePremiumOCR",
    "LlamaParseLVMOCR",
    "LlamaParseLayoutAgentOCR",
    "MistralOCR",
    "GeminiOCR",
    "MODEL_REGISTRY",
]
