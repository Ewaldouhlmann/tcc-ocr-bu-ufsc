from abc import ABC, abstractmethod
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class BaseOCR(ABC):
    """Interface comum para todos os modelos OCR do projeto."""

    #: Nome curto que identifica o modelo nos resultados do benchmark.
    name: str = "base"

    @abstractmethod
    def recognize(self, image_path: Path) -> str:
        """Recebe o caminho de uma imagem PNG e retorna o texto extraído."""
        ...

    def batch_recognize(self, image_paths: list[Path]) -> list[str]:
        """Processa uma lista de imagens sequencialmente.
        Pode ser sobrescrito por subclasses para processamento em lote otimizado."""
        return [self.recognize(p) for p in image_paths]


class ClassicalOCR(BaseOCR):
    """Motores OCR clássicos e baseados em CRNN (Tesseract, EasyOCR, PaddleOCR, DocTR).

    Operam localmente em CPU (com aceleração opcional por GPU) e não dependem de
    conhecimento linguístico pré-treinado em larga escala.
    """


class TransformerOCR(BaseOCR):
    """Modelos encoder-decoder baseados em Transformer (TrOCR), via Hugging Face.

    Diferentemente dos VLMs, operam sobre linhas de texto já segmentadas: cada
    imagem de página é primeiro dividida em linhas com o CRAFT
    (`craft-text-detector`) e cada linha é reconhecida individualmente.
    """

    #: Checkpoint do Hugging Face Hub a ser carregado; definido pelas subclasses.
    checkpoint: str = ""

    def __init__(self, device: str | None = None):
        import torch
        from craft_text_detector import Craft
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(self.checkpoint)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.checkpoint).to(self.device)
        self.craft = Craft(output_dir=None, crop_type="box", cuda=(self.device == "cuda"))

    def recognize(self, image_path: Path) -> str:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        prediction = self.craft.detect_text(str(image_path))

        # Ordena as linhas de cima para baixo, seguindo a ordem de leitura da página.
        boxes = sorted(prediction["boxes"], key=lambda box: min(point[1] for point in box))

        lines = []
        for box in boxes:
            xs = [point[0] for point in box]
            ys = [point[1] for point in box]
            crop = image.crop((min(xs), min(ys), max(xs), max(ys)))
            lines.append(self._recognize_line(crop))
        return "\n".join(lines)

    def _recognize_line(self, line_image) -> str:
        pixel_values = self.processor(images=line_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


class VLMOCR(BaseOCR):
    """Modelos de visão-linguagem (VLMs), executados localmente via Ollama ou via API.

    Recebem a página completa, sem segmentação, e geram a transcrição por meio de
    um prompt instrucional padronizado, idêntico para todos os modelos desta
    categoria (conforme Seção 5.2 do TCC).
    """

    PROMPT = (
        "Transcreva fielmente todo o texto visível nesta página, preservando a "
        "ortografia, a acentuação e a pontuação originais, mesmo que arcaicas ou "
        "não padronizadas. Não modernize grafias históricas. Ignore elementos não "
        "textuais, como numeração de página, ornamentos tipográficos e marcas "
        "d'água. Responda apenas com o texto transcrito, sem comentários "
        "adicionais."
    )
