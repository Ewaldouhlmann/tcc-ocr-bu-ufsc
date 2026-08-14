from src.ocr.vlm._ollama_base import OllamaVLM


class GemmaOCR(OllamaVLM):
    """Gemma 4 (Google DeepMind), via Ollama.

    Disponível em quatro tamanhos (E2B, E4B, 26B, 31B); usa-se a menor variante
    compatível com a VRAM disponível (Seção 5.1.2 do TCC). Confirme a tag exata
    publicada no Ollama antes de usar (`ollama pull gemma4`).
    """

    name = "gemma4"
    model_tag = "gemma4:2b"
