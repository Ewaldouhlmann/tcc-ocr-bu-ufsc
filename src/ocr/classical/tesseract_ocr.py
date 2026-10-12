from pathlib import Path

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
        # Import local, como nos demais adaptadores: carregar src.ocr não deve
        # exigir as dependências de todos os modelos (ex.: no Colab).
        import pytesseract
        from PIL import Image

        image = Image.open(image_path)
        return pytesseract.image_to_string(image, lang=self.lang, config=self.config)
