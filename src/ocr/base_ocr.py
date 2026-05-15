from abc import ABC, abstractmethod
from pathlib import Path


class BaseOCR(ABC):
    """Interface comum para todos os modelos OCR do projeto."""

    @abstractmethod
    def recognize(self, image_path: Path) -> str:
        """Recebe o caminho de uma imagem PNG e retorna o texto extraído."""
        ...

    def batch_recognize(self, image_paths: list[Path]) -> list[str]:
        """Processa uma lista de imagens sequencialmente.
        Pode ser sobrescrito por subclasses para processamento em lote otimizado."""
        return [self.recognize(p) for p in image_paths]
