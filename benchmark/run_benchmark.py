"""Ponto de entrada CLI para execução do benchmark comparativo."""

import argparse
from pathlib import Path


AVAILABLE_MODELS = [
    "tesseract", "easyocr", "paddleocr", "doctr",
    "trocr-base", "trocr-large",
    "gemma4", "deepseek", "qwen",
    "llamaparse-free", "llamaparse-premium",
    "mistral", "gemini",
]


def main():
    parser = argparse.ArgumentParser(description="Benchmark comparativo de modelos OCR.")
    parser.add_argument("--images", required=True, help="Pasta com imagens de amostra (PNG)")
    parser.add_argument("--ground-truth", required=True, help="Pasta com transcrições de referência")
    parser.add_argument("--models", nargs="+", choices=AVAILABLE_MODELS + ["all"],
                        default=["all"], help="Modelos a avaliar")
    parser.add_argument("--output", required=True, help="Pasta para salvar resultados")
    args = parser.parse_args()

    models = AVAILABLE_MODELS if "all" in args.models else args.models
    print(f"Modelos selecionados: {models}")
    # TODO: implementar loop de benchmark


if __name__ == "__main__":
    main()
