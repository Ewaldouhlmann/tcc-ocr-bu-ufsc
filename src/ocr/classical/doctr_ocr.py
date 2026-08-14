from pathlib import Path

from src.ocr.base_ocr import ClassicalOCR


class DocTROCR(ClassicalOCR):
    """DocTR (detector DBNet/LinkNet + reconhecedor CRNN/SAR), backend PyTorch.

    Instanciado com os modelos padrão de detecção (db_resnet50) e reconhecimento
    (crnn_vgg16_bn), aplicados diretamente sobre a imagem PNG, conforme
    Seção 5.1.2 do TCC.
    """

    name = "doctr"

    def __init__(
        self,
        det_arch: str = "db_resnet50",
        reco_arch: str = "crnn_vgg16_bn",
        pretrained: bool = True,
    ):
        from doctr.models import ocr_predictor

        self.model = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=pretrained)

    def recognize(self, image_path: Path) -> str:
        from doctr.io import DocumentFile

        doc = DocumentFile.from_images(str(image_path))
        result = self.model(doc)

        lines = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    lines.append(" ".join(word.value for word in line.words))
        return "\n".join(lines)
