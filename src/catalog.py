"""Carrega o catálogo de obras e modelos (data/catalog.yaml).

Fonte única de metadados usada pelo pipeline (converter.py, extractor.py) e
pelo benchmark, evitando duplicar título/autor/modelo em vários arquivos.
"""

from pathlib import Path

import yaml

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.yaml"


def load_catalog(path: Path = CATALOG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_obras(path: Path = CATALOG_PATH) -> dict[str, dict]:
    """Retorna {obra_slug: {titulo, autor, ano, total_paginas, tipografia, pdf}}."""
    return load_catalog(path)["obras"]


def load_modelos(path: Path = CATALOG_PATH) -> dict[str, dict]:
    """Retorna {model_name: {nome, categoria, arquivo, custo, gpu, ...}}, com as
    mesmas chaves de src/ocr/__init__.py::MODEL_REGISTRY."""
    return load_catalog(path)["modelos"]
