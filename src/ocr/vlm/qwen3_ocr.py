from src.ocr.vlm._ollama_base import OllamaVLM


class Qwen3OCR(OllamaVLM):
    """Qwen3-VL 2B (Alibaba), via Ollama.

    Sucessor do Qwen2.5-VL, na menor variante disponível (1,9 GB, q4_K_M).
    Avaliado como alternativa local ao Qwen2.5-VL 3B, que excede a memória do
    hardware do projeto em páginas degradadas.
    """

    name = "qwen3-vl"
    model_tag = "qwen3-vl:2b"
    # Com o raciocínio ativo (padrão da tag), o modelo gastava toda a janela de
    # contexto "pensando" e devolvia a transcrição vazia.
    think = False
