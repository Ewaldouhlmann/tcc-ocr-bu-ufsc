import base64
import os
from pathlib import Path

from src.ocr.base_ocr import VLMOCR


class MistralOCR(VLMOCR):
    """Mistral OCR, via API REST (biblioteca mistralai).

    A imagem é enviada codificada em base64; a resposta contém o texto
    transcrito em formato estruturado (Markdown por página), conforme
    Seção 5.1.2 do TCC. Autenticação por MISTRAL_API_KEY (ver .env.example).
    """

    name = "mistral"

    def __init__(self, api_key: str | None = None, model: str = "mistral-ocr-latest"):
        from mistralai import Mistral

        self._client = Mistral(api_key=api_key or os.getenv("MISTRAL_API_KEY"))
        self.model = model

    def recognize(self, image_path: Path) -> str:
        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        response = self._client.ocr.process(
            model=self.model,
            document={
                "type": "image_url",
                "image_url": f"data:image/png;base64,{image_b64}",
            },
        )
        return "\n\n".join(page.markdown for page in response.pages)
