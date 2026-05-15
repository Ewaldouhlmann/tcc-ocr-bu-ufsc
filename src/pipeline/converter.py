"""Converte páginas de PDF em imagens PNG para uso no pipeline OCR."""

from pathlib import Path
import argparse


def convert_pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 300) -> list[Path]:
    """Converte um PDF em uma lista de imagens PNG, uma por página."""
    # TODO: implementar com pdf2image
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="Converte PDFs em imagens PNG.")
    parser.add_argument("--input", required=True, help="Pasta com os PDFs de entrada")
    parser.add_argument("--output", required=True, help="Pasta de saída para os PNGs")
    parser.add_argument("--dpi", type=int, default=300, help="Resolução em DPI (padrão: 300)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf in sorted(input_dir.glob("*.pdf")):
        print(f"Convertendo {pdf.name}...")
        convert_pdf_to_images(pdf, output_dir / pdf.stem, dpi=args.dpi)


if __name__ == "__main__":
    main()
