"""VLMs locais que exigem GPU, executados via Ollama em ambiente de nuvem
(Google Colab/Kaggle, GPU T4 com 15 GB).

Não cabem no equipamento do trabalho (7,5 GB de RAM, sem GPU), mas entram no
benchmark de qualidade como referência. Ver notebooks/benchmark_gpu.ipynb.
"""

from src.ocr.vlm._ollama_base import OllamaVLM


class Qwen7BOCR(OllamaVLM):
    """Qwen2.5-VL na variante de 7B, a avaliada por Vesalainen et al. (2025)."""

    name = "qwen-7b"
    model_tag = "qwen2.5vl:7b"


class Gemma4E4BOCR(OllamaVLM):
    """Gemma 4 E4B (Google DeepMind), 9,6 GB no Ollama."""

    name = "gemma4-e4b"
    model_tag = "gemma4:e4b"


class DeepSeekOCRModel(OllamaVLM):
    """DeepSeek-OCR 3B, modelo da família DeepSeek disponível no Ollama.

    Treinado com instruções de tarefa curtas; "Free OCR." pede a transcrição
    do texto sem formatação adicional.
    """

    name = "deepseek-ocr"
    model_tag = "deepseek-ocr:3b"
    PROMPT = "Free OCR."


class PaddleOCRVLOCR(OllamaVLM):
    """PaddleOCR-VL 0.9B (Baidu), versão comunitária no Ollama.

    Usa os prompts de tarefa do modelo; "OCR:" é o de reconhecimento de texto.
    """

    name = "paddleocr-vl"
    model_tag = "MedAIBase/PaddleOCR-VL:0.9b"
    PROMPT = "OCR:"
