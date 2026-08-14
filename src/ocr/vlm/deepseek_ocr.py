from src.ocr.vlm._ollama_base import OllamaVLM


class DeepSeekOCR(OllamaVLM):
    """DeepSeek-VL2 (mixture-of-experts), via Ollama.

    Disponível em variantes de 1,0B, 2,8B e 4,5B de parâmetros ativos; usa-se a
    menor variante compatível com a VRAM disponível (Seção 5.1.2 do TCC).
    Confirme a tag exata publicada no Ollama antes de usar
    (`ollama pull deepseek-vl2`).
    """

    name = "deepseek"
    model_tag = "deepseek-vl2:1b"
