import os
from pathlib import Path

from src.ocr.base_ocr import VLMOCR


class GeminiOCR(VLMOCR):
    """Gemini 2.0 Flash (Google), via google-generativeai.

    A imagem é enviada diretamente como parte do conteúdo multimodal da
    requisição, junto ao prompt padronizado de VLMOCR, conforme Seção 5.1.2 do
    TCC. Autenticação por GEMINI_API_KEY (ver .env.example).

    Descontinuação anunciada para jun./2026; se necessário, substituir por
    "gemini-2.5-flash" no parâmetro model_name.
    """

    name = "gemini"

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-2.0-flash"):
        import google.generativeai as genai

        genai.configure(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        self._model = genai.GenerativeModel(model_name)

    def recognize(self, image_path: Path) -> str:
        from PIL import Image

        image = Image.open(image_path)
        response = self._model.generate_content([self.PROMPT, image])
        return response.text
