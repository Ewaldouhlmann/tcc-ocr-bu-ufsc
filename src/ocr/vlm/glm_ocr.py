from src.ocr.vlm._ollama_base import OllamaVLM


class GLMOCR(OllamaVLM):
    """GLM-OCR (Zhipu AI, 0,9B parâmetros), via Ollama.

    Modelo especializado em OCR de documentos, leve o bastante para rodar em CPU
    com pouca memória (1,6 GB na quantização q8_0). Incluído após a verificação
    de que Gemma 4 e DeepSeek-VL2 não cabem/não existem no Ollama para o
    hardware do projeto.

    Diferente dos VLMs de propósito geral, o GLM-OCR foi treinado com prompts
    de tarefa fixos; o prompt instrucional em português de VLMOCR não se aplica.
    """

    name = "glm-ocr"
    model_tag = "glm-ocr:q8_0"
    PROMPT = "Text Recognition:"
