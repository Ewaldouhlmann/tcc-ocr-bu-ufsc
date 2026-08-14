from pathlib import Path

import pytesseract
from PIL import Image

from src.ocr.base_ocr import ClassicalOCR


class TesseractOCR(ClassicalOCR):
    """Tesseract 5, via pytesseract.

    Configurado com PSM 6 (bloco de texto uniforme) e idiomas por+eng, conforme
    Seção 5.1.2 do TCC.
    """

    name = "tesseract"

    def __init__(self, lang: str = "por+eng", psm: int = 6):
        self.lang = lang
        self.config = f"--psm {psm}"

    def recognize(self, image_path: Path) -> str:
        image = Image.open(image_path)
        return pytesseract.image_to_string(image, lang=self.lang, config=self.config)
