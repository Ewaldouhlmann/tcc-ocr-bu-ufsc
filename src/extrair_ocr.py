"""
Extração OCR de PDF escaneado
==============================
Uso:
    python src/extrair_ocr.py input/arquivo.pdf
    python src/extrair_ocr.py input/arquivo.pdf --modelo easyocr
    python src/extrair_ocr.py input/arquivo.pdf --modelo kraken
    python src/extrair_ocr.py input/arquivo.pdf --modelo paddleocr

Modelos disponíveis:
    tesseract  (padrão) — clássico, leve, bom suporte a português
    easyocr             — fácil de instalar, sem dependências externas
    kraken              — especializado em documentos históricos
    paddleocr           — estado da arte em benchmarks recentes

Saída (pasta output/nome_do_arquivo/modelo/):
    - pagina_001.txt, pagina_002.txt ...   texto de cada página
    - pagina_001.json, pagina_002.json ... palavras com posição e confiança
    - resultado_completo.txt               texto de todas as páginas junto
"""

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

# ─────────────────────────────────────────────
# RASTERIZAÇÃO
# ─────────────────────────────────────────────


def rasterizar_paginas(pdf_path: Path, dpi: int = 300) -> list[Path]:
    """Converte todas as páginas do PDF em imagens PNG."""
    tmp_dir = Path(tempfile.mkdtemp())

    print(f"Convertendo PDF para imagens a {dpi} DPI...")

    if platform.system() == "Windows":
        from pdf2image import convert_from_path

        paginas = convert_from_path(str(pdf_path), dpi=dpi)
        imagens = []
        for i, pag in enumerate(paginas, start=1):
            img_path = tmp_dir / f"pagina-{i:03d}.png"
            pag.save(img_path, "PNG")
            imagens.append(img_path)
    else:
        subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-r",
                str(dpi),
                str(pdf_path),
                str(tmp_dir / "pagina"),
            ],
            check=True,
        )
        imagens = sorted(tmp_dir.glob("pagina-*.png"))

    print(f"  {len(imagens)} páginas encontradas.")
    return imagens


# ─────────────────────────────────────────────
# MODELOS
# ─────────────────────────────────────────────


def extrair_tesseract(imagem: Path) -> dict:
    import pytesseract
    from PIL import Image

    if platform.system() == "Windows":
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

    img = Image.open(imagem)
    texto = pytesseract.image_to_string(img, lang="por", config="--oem 3 --psm 3")
    data = pytesseract.image_to_data(
        img, lang="por", config="--oem 3 --psm 3", output_type=pytesseract.Output.DICT
    )

    palavras = []
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if not word:
            continue
        palavras.append(
            {
                "texto": word,
                "x": data["left"][i],
                "y": data["top"][i],
                "largura": data["width"][i],
                "altura": data["height"][i],
                "confianca": int(data["conf"][i]),
            }
        )

    return {"texto": texto.strip(), "palavras": palavras}


def extrair_easyocr(imagem: Path) -> dict:
    import easyocr

    reader = easyocr.Reader(["pt"], gpu=False)
    resultados = reader.readtext(str(imagem))

    palavras = []
    linhas = []
    for bbox, texto, confianca in resultados:
        x = int(bbox[0][0])
        y = int(bbox[0][1])
        largura = int(bbox[2][0] - bbox[0][0])
        altura = int(bbox[2][1] - bbox[0][1])
        palavras.append(
            {
                "texto": texto,
                "x": x,
                "y": y,
                "largura": largura,
                "altura": altura,
                "confianca": round(confianca * 100, 1),
            }
        )
        linhas.append(texto)

    return {"texto": "\n".join(linhas), "palavras": palavras}


def extrair_kraken(imagem: Path) -> dict:
    from kraken import binarization, pageseg, rpred
    from kraken.lib import models
    from PIL import Image

    img = Image.open(imagem).convert("L")
    bin_img = binarization.nlbin(img)
    baseline_seg = pageseg.segment(bin_img)

    # Substitua pelo caminho de um modelo treinado para português histórico
    # Modelos disponíveis em: https://zenodo.org/communities/ocr_models
    model = models.load_any("en-default.mlmodel")

    palavras = []
    linhas_texto = []
    for line in rpred.rpred(model, bin_img, baseline_seg):
        texto = line.prediction
        if texto.strip():
            linhas_texto.append(texto.strip())
            palavras.append(
                {
                    "texto": texto.strip(),
                    "x": int(line.bbox[0]),
                    "y": int(line.bbox[1]),
                    "largura": int(line.bbox[2] - line.bbox[0]),
                    "altura": int(line.bbox[3] - line.bbox[1]),
                    "confianca": None,
                }
            )

    return {"texto": "\n".join(linhas_texto), "palavras": palavras}


def extrair_paddleocr(imagem: Path) -> dict:
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=True, lang="pt", show_log=False)
    resultado = ocr.ocr(str(imagem), cls=True)

    palavras = []
    linhas = []
    if resultado and resultado[0]:
        for linha in resultado[0]:
            bbox, (texto, confianca) = linha
            x = int(bbox[0][0])
            y = int(bbox[0][1])
            largura = int(bbox[2][0] - bbox[0][0])
            altura = int(bbox[2][1] - bbox[0][1])
            palavras.append(
                {
                    "texto": texto,
                    "x": x,
                    "y": y,
                    "largura": largura,
                    "altura": altura,
                    "confianca": round(confianca * 100, 1),
                }
            )
            linhas.append(texto)

    return {"texto": "\n".join(linhas), "palavras": palavras}


# ─────────────────────────────────────────────
# DESPACHANTE
# ─────────────────────────────────────────────

MODELOS = {
    "tesseract": extrair_tesseract,
    "easyocr": extrair_easyocr,
    "kraken": extrair_kraken,
    "paddleocr": extrair_paddleocr,
}


def extrair_pagina(imagem: Path, modelo: str) -> dict:
    if modelo not in MODELOS:
        print(f"Modelo '{modelo}' desconhecido. Opções: {', '.join(MODELOS)}")
        sys.exit(1)
    return MODELOS[modelo](imagem)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Extração OCR de PDF escaneado")
    parser.add_argument("pdf", help="Caminho para o PDF escaneado")
    parser.add_argument(
        "--modelo",
        default="tesseract",
        choices=MODELOS.keys(),
        help="Motor de OCR a usar (padrão: tesseract)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolução para rasterizar as páginas (padrão: 300)",
    )
    parser.add_argument(
        "--pagina",
        type=int,
        default=None,
        help="Processar apenas uma página específica (ex: --pagina 5)",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Arquivo não encontrado: {pdf_path}")
        sys.exit(1)

    output_dir = Path("output") / pdf_path.stem / args.modelo
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Modelo: {args.modelo}")
    print(f"Saída:  {output_dir.resolve()}\n")

    imagens = rasterizar_paginas(pdf_path, dpi=args.dpi)

    # Filtra para uma página específica se solicitado
    if args.pagina is not None:
        if args.pagina < 1 or args.pagina > len(imagens):
            print(f"Página {args.pagina} inválida. O PDF tem {len(imagens)} páginas.")
            sys.exit(1)
        imagens = [imagens[args.pagina - 1]]
        print(f"Processando apenas a página {args.pagina}.\n")

    texto_completo = []

    # Define o número inicial da página (relevante quando --pagina é usado)
    pagina_inicial = args.pagina if args.pagina is not None else 1

    for i, imagem in enumerate(imagens, start=pagina_inicial):
        print(f"Processando página {i}/{len(imagens)}...", end=" ", flush=True)

        resultado = extrair_pagina(imagem, args.modelo)

        (output_dir / f"pagina_{i:03d}.txt").write_text(
            resultado["texto"], encoding="utf-8"
        )
        (output_dir / f"pagina_{i:03d}.json").write_text(
            json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        texto_completo.append(f"=== Página {i} ===\n{resultado['texto']}")
        print(f"{len(resultado['palavras'])} palavras.")

    (output_dir / "resultado_completo.txt").write_text(
        "\n\n".join(texto_completo), encoding="utf-8"
    )

    print(f"\nPronto! {len(imagens)} páginas salvas em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
