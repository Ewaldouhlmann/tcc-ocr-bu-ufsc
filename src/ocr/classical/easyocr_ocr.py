from pathlib import Path

from src.ocr.base_ocr import ClassicalOCR


class EasyOCR(ClassicalOCR):
    """EasyOCR (detecção CRAFT + reconhecimento CRNN).

    Inicializado com os idiomas ['pt', 'en'] e GPU habilitada, conforme
    Seção 5.1.2 do TCC.
    """

    name = "easyocr"

    def __init__(self, languages: list[str] | None = None, gpu: bool = True):
        import easyocr

        self.reader = easyocr.Reader(languages or ["pt", "en"], gpu=gpu)

    def recognize(self, image_path: Path) -> str:
        lines = self.reader.readtext(str(image_path), detail=0, paragraph=True)
        return "\n".join(lines)
