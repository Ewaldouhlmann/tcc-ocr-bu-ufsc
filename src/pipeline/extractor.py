"""Pipeline de extração: aplica um modelo OCR a um livro inteiro (PDF).

Para cada página: renderiza a página do PDF em PNG, passa a imagem ao modelo
escolhido e salva o texto. As páginas são processadas uma a uma, sem converter
o livro todo antes, para não ocupar disco nem memória à toa.

Saídas em --output/{nome_do_pdf}/:
- paginas/{pagina:04d}.txt   texto de cada página
- execucao.csv               tempo e tokens por página
- falhas.csv                 páginas em que o modelo falhou (se houver)
- {nome_do_pdf}.txt          livro completo, páginas na ordem, com marcadores

Rodar de novo retoma de onde parou: páginas já salvas são puladas.

Uso:
    python -m src.pipeline.extractor --pdf livro.pdf --model glm-ocr --output saida/
    python -m src.pipeline.extractor --pdf livro.pdf --model qwen-7b --output saida/ --paginas 1-20,35
"""

import argparse
import csv
import tempfile
import time
from pathlib import Path

from pdf2image import convert_from_path, pdfinfo_from_path


def parse_paginas(texto: str | None, total: int) -> list[int]:
    """'1-10,15' -> [1, ..., 10, 15]. Vazio = todas as páginas do PDF."""
    if not texto:
        return list(range(1, total + 1))
    paginas = set()
    for parte in texto.split(","):
        inicio, _, fim = parte.strip().partition("-")
        paginas.update(range(int(inicio), int(fim or inicio) + 1))
    return sorted(p for p in paginas if 1 <= p <= total)


def juntar_livro(pasta_paginas: Path, destino: Path):
    """Concatena as páginas salvas em um único arquivo, com marcador de página."""
    partes = [f"===== Página {int(arq.stem)} =====\n{arq.read_text(encoding='utf-8').strip()}\n"
              for arq in sorted(pasta_paginas.glob("*.txt"))]
    destino.write_text("\n".join(partes), encoding="utf-8")


def run_extraction(pdf_path: Path, model_name: str, output_dir: Path,
                   paginas: str | None = None, dpi: int = 300) -> Path:
    """Extrai o texto das páginas do PDF com o modelo indicado e devolve o
    caminho do arquivo com o livro completo."""
    from src.ocr import MODEL_REGISTRY

    if model_name not in MODEL_REGISTRY:
        raise SystemExit(f"Modelo desconhecido: {model_name}. Opções: {', '.join(MODEL_REGISTRY)}")

    total = pdfinfo_from_path(str(pdf_path))["Pages"]
    selecionadas = parse_paginas(paginas, total)
    destino = output_dir / pdf_path.stem
    pasta_paginas = destino / "paginas"
    pasta_paginas.mkdir(parents=True, exist_ok=True)
    pendentes = [p for p in selecionadas if not (pasta_paginas / f"{p:04d}.txt").exists()]
    print(f"{pdf_path.name}: {total} páginas no PDF, {len(selecionadas)} selecionadas, "
          f"{len(pendentes)} pendentes. Modelo: {model_name}", flush=True)

    modelo = MODEL_REGISTRY[model_name]() if pendentes else None
    registro = destino / "execucao.csv"
    novo = not registro.exists()
    with open(registro, "a", newline="", encoding="utf-8") as f, tempfile.TemporaryDirectory() as tmp:
        escritor = csv.writer(f)
        if novo:
            escritor.writerow(["pagina", "modelo", "tempo_s", "caracteres", "tokens"])
        for i, pagina in enumerate(pendentes, start=1):
            imagem = Path(tmp) / f"{pagina:04d}.png"
            convert_from_path(str(pdf_path), dpi=dpi, first_page=pagina,
                              last_page=pagina)[0].save(imagem, "PNG")
            inicio = time.perf_counter()
            try:
                texto = modelo.recognize(imagem)
            except Exception as e:  # uma página com problema não interrompe o livro
                with open(destino / "falhas.csv", "a", encoding="utf-8") as falhas:
                    falhas.write(f"{pagina},{model_name},{type(e).__name__}: {str(e)[:200]!r}\n")
                print(f"  [{i}/{len(pendentes)}] página {pagina}: ERRO {type(e).__name__}: {e}", flush=True)
                continue
            finally:
                imagem.unlink(missing_ok=True)
            tempo = time.perf_counter() - inicio
            (pasta_paginas / f"{pagina:04d}.txt").write_text(texto, encoding="utf-8")
            escritor.writerow([pagina, model_name, round(tempo, 2), len(texto),
                               getattr(modelo, "last_tokens_used", None)])
            f.flush()
            print(f"  [{i}/{len(pendentes)}] página {pagina}: {len(texto)} caracteres em {tempo:.1f} s",
                  flush=True)

    livro = destino / f"{pdf_path.stem}.txt"
    juntar_livro(pasta_paginas, livro)
    feitas = len(list(pasta_paginas.glob("*.txt")))
    print(f"Concluído: {feitas} páginas salvas. Livro completo em {livro}", flush=True)
    return livro


def main():
    parser = argparse.ArgumentParser(description="Pipeline de extração OCR de um livro em PDF.")
    parser.add_argument("--pdf", required=True, help="Arquivo PDF do livro")
    parser.add_argument("--model", required=True, help="Nome do modelo (chave do MODEL_REGISTRY)")
    parser.add_argument("--output", required=True, help="Pasta de saída")
    parser.add_argument("--paginas", default=None, help="Páginas a processar, ex.: 1-20,35 (padrão: todas)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Resolução da renderização (padrão: 300). Digitalizações de "
                             "~150 ppi não ganham detalhe acima disso, só custo.")
    args = parser.parse_args()
    run_extraction(Path(args.pdf), args.model, Path(args.output), args.paginas, args.dpi)


if __name__ == "__main__":
    main()
