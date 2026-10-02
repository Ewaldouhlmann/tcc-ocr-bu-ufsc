import os
from pathlib import Path

from src.ocr.base_ocr import VLMOCR


class GeminiOCR(VLMOCR):
    """Gemini Flash (Google), via google-generativeai.

    A imagem é enviada diretamente como parte do conteúdo multimodal da
    requisição, junto ao prompt padronizado de VLMOCR, conforme Seção 5.1.2 do
    TCC. Autenticação por GEMINI_API_KEY (ver .env.example).

    O Gemini 2.0 Flash, previsto originalmente no TCC, foi descontinuado em
    jun./2026 e o 2.5 Flash não está disponível para novas contas; por isso o
    padrão é o gemini-3.8-flash (verificado em 25/09/2026).
    """

    name = "gemini"

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-3.8-flash"):
        import google.generativeai as genai

        genai.configure(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        self._model = genai.GenerativeModel(model_name)

    def recognize(self, image_path: Path) -> str:
        from PIL import Image

        image = Image.open(image_path)
        response = self._model.generate_content([self.PROMPT, image])

        try:
            self.last_tokens_used = response.usage_metadata.total_token_count
        except AttributeError:
            self.last_tokens_used = None

        return response.text
