from src.ocr.vlm._ollama_base import OllamaVLM


class QwenOCR(OllamaVLM):
    """Qwen2.5-VL (Alibaba), via Ollama.

    Modelo selecionado ao final da Fase 2 do TCC (Seção 5.2.3) para compor o
    pipeline de extração completa, por apresentar os melhores CER/WER entre os
    modelos gratuitos e por ser projetado para documentos de layout complexo.
    No pipeline final (Fase 3), os resultados são salvos como
    `data/results/{livro}/{pagina}/qwen25vl.txt`.
    """

    name = "qwen"
    model_tag = "qwen2.5vl:3b"
