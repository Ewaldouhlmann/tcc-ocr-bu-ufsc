from src.ocr.base_ocr import TransformerOCR


class TrOCRLarge(TransformerOCR):
    """TrOCR large-handwritten (Microsoft), via Hugging Face.

    Eliminado na Fase 2 do TCC por custo computacional (Seção 5.2.1), mas
    mantido aqui para compor o benchmark comparativo da Fase 1.
    """

    name = "trocr-large"
    checkpoint = "microsoft/trocr-large-handwritten"
