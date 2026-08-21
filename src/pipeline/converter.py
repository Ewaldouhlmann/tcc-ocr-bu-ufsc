"""Converte páginas de PDF em imagens PNG para uso no pipeline OCR."""

import csv
import os
from pathlib import Path
import argparse

from pdf2image import convert_from_path

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent.parent.parent / "benchmark" / "ground_truth" / "manifest.csv"
)


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


def convert_pdf_pages(pdf_path: Path, output_dir: Path, pages: list[int], dpi: int = 300) -> list[Path]:
    """Converte só as páginas indicadas de um PDF (numeração 1-based), sem
    renderizar o restante do livro.

    Útil para preparar o corpus de amostra do benchmark a partir de obras
    longas (ex.: "O Feliz Independente", 1043 páginas, das quais só 3 entram
    no manifest.csv), evitando converter o livro inteiro à toa. Cada página é
    salva como "{pagina:03d}.png" dentro de output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for page_number in pages:
        rendered = convert_from_path(str(pdf_path), dpi=dpi, first_page=page_number, last_page=page_number)
        image_path = output_dir / f"{page_number:03d}.png"
        rendered[0].save(image_path, "PNG")
        image_paths.append(image_path)
    return image_paths


def _load_manifest_pages(manifest_path: Path) -> dict[str, list[int]]:
    """Lê manifest.csv e retorna {obra_slug: [páginas]}, para a conversão seletiva."""
    pages_by_obra: dict[str, list[int]] = {}
    with open(manifest_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pages_by_obra.setdefault(row["obra_slug"], []).append(int(row["pagina"]))
    return pages_by_obra


def main():
    parser = argparse.ArgumentParser(description="Converte PDFs em imagens PNG.")
    parser.add_argument("--input", required=True, help="Pasta com os PDFs de entrada")
    parser.add_argument("--output", required=True, help="Pasta de saída para os PNGs")
    parser.add_argument("--dpi", type=int, default=300, help="Resolução em DPI (padrão: 300)")
    parser.add_argument(
        "--manifest", nargs="?", const=str(DEFAULT_MANIFEST), default=None,
        help="Converte só as páginas listadas no CSV obra_slug/pagina (em vez do livro "
             f"inteiro); cada PDF deve se chamar {{obra_slug}}.pdf. Sem valor, usa "
             f"{DEFAULT_MANIFEST}",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        pages_by_obra = _load_manifest_pages(Path(args.manifest))
        for obra_slug, pages in sorted(pages_by_obra.items()):
            pdf_path = input_dir / f"{obra_slug}.pdf"
            if not pdf_path.exists():
                print(f"Aviso: {pdf_path} não encontrado, pulando {obra_slug}")
                continue
            print(f"Convertendo {pdf_path.name} (páginas {sorted(pages)})...")
            convert_pdf_pages(pdf_path, output_dir / obra_slug, sorted(pages), dpi=args.dpi)
    else:
        for pdf in sorted(input_dir.glob("*.pdf")):
            print(f"Convertendo {pdf.name}...")
            convert_pdf_to_images(pdf, output_dir / pdf.stem, dpi=args.dpi)


if __name__ == "__main__":
    main()
