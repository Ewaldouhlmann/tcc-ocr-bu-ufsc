from pathlib import Path

from src.ocr.base_ocr import VLMOCR


class OllamaVLM(VLMOCR):
    """Base para VLMs executados localmente via Ollama.

    A interação ocorre pela API REST local do Ollama (http://localhost:11434 por
    padrão), enviando a imagem da página e o prompt padronizado de VLMOCR,
    conforme Seção 5.1.2 do TCC.
    """

    #: Tag do modelo no Ollama (ex.: "qwen2.5vl:3b"); definida pelas subclasses.
    model_tag: str = ""

    def __init__(self, model_tag: str | None = None):
        import ollama

        self._client = ollama
        if model_tag:
            self.model_tag = model_tag

    def recognize(self, image_path: Path) -> str:
        response = self._client.chat(
            model=self.model_tag,
            messages=[
                {
                    "role": "user",
                    "content": self.PROMPT,
                    "images": [str(image_path)],
                }
            ],
        )

        try:
            self.last_tokens_used = response["prompt_eval_count"] + response["eval_count"]
        except (KeyError, TypeError):
            self.last_tokens_used = None

        return response["message"]["content"]
