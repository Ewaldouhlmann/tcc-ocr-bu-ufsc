"""Converte páginas de PDF em imagens PNG para uso no pipeline OCR."""

import os
from pathlib import Path
import argparse

from pdf2image import convert_from_path


def convert_pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 300) -> list[Path]:
    """Converte um PDF em uma lista de imagens PNG, uma por página.

    Cada página é salva como "{numero_da_pagina:03d}.png" dentro de output_dir,
    numerada a partir de 1, seguindo a resolução de 300 DPI descrita na
    Seção 5.3 do TCC.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pages = convert_from_path(str(pdf_path), dpi=dpi, thread_count=os.cpu_count() or 1)

    image_paths = []
    for page_number, page in enumerate(pages, start=1):
        image_path = output_dir / f"{page_number:03d}.png"
        page.save(image_path, "PNG")
        image_paths.append(image_path)
    return image_paths


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
