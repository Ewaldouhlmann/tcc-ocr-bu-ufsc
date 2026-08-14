from pathlib import Path

from src.ocr.base_ocr import ClassicalOCR


class PaddleOCR(ClassicalOCR):
    """PaddleOCR (detecção + classificação de orientação + reconhecimento).

    Configurado com lang='latin', a opção mais próxima disponível para o
    português no modelo de reconhecimento padrão, conforme Seção 5.1.2 do TCC.
    """

    name = "paddleocr"

    def __init__(self, lang: str = "latin", use_gpu: bool = False):
        from paddleocr import PaddleOCR as _PaddleOCREngine

        self.engine = _PaddleOCREngine(lang=lang, use_angle_cls=True, use_gpu=use_gpu, show_log=False)

    def recognize(self, image_path: Path) -> str:
        result = self.engine.ocr(str(image_path), cls=True)
        lines = [line[1][0] for page in result for line in page]
        return "\n".join(lines)
