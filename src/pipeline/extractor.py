"""Orquestra o pipeline completo de extração de texto para o corpus."""

from pathlib import Path
import argparse


def run_extraction(corpus_dir: Path, model_name: str, output_dir: Path):
    """Aplica o modelo selecionado a todo o corpus e salva os resultados."""
    # TODO: implementar
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="Pipeline de extração OCR para o corpus.")
    parser.add_argument("--corpus", required=True, help="Pasta raiz do corpus")
    parser.add_argument("--model", required=True, help="Nome do modelo OCR a usar")
    parser.add_argument("--output", required=True, help="Pasta de saída dos resultados")
    args = parser.parse_args()

    run_extraction(Path(args.corpus), args.model, Path(args.output))


if __name__ == "__main__":
    main()
