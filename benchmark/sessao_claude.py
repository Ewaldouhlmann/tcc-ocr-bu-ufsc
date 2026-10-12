"""Transcrição das páginas pelo Claude em uma sessão isolada do Claude Code.

Alternativa à extração via API (src/ocr/vlm/claude_ocr.py) quando não há
créditos de API. A regra ouro partiu de uma transcrição do Claude; para que a
nova transcrição não seja contaminada por ela, a sessão roda numa pasta fora
do repositório, com acesso apenas às imagens e ao prompt padronizado.

Uso:
    python benchmark/sessao_claude.py preparar   # cria ~/claude_ocr_isolado
    # abrir uma sessão NOVA do Claude Code dentro de ~/claude_ocr_isolado
    # e seguir INSTRUCOES.md
    python benchmark/sessao_claude.py coletar    # copia as saídas para o benchmark

Esta rodada não tem tempo, CPU, RAM nem tokens medidos: só o texto é usado,
para o CER/WER.
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src.ocr.base_ocr import VLMOCR  # noqa: E402

MANIFESTO = RAIZ / "data" / "corpus" / "regra_ouro" / "manifest.csv"
IMAGENS = RAIZ / "data" / "corpus" / "images"
SAIDA_BENCHMARK = RAIZ / "data" / "results" / "extracao" / "claude"
PASTA_ISOLADA = Path.home() / "claude_ocr_isolado"

INSTRUCOES = """# Transcrição de páginas

Para cada imagem em `imagens/{{obra}}/{{pagina}}.png`, crie
`saidas/{{obra}}/{{pagina}}.txt` com a transcrição da página.

Regras:

- Leia cada imagem diretamente (ferramenta Read). Não use Tesseract, OCR ou
  qualquer outro programa de reconhecimento de texto.
- Trate cada página de forma independente, seguindo apenas a instrução abaixo.
- Não procure nem leia outros arquivos fora desta pasta.
- O arquivo de saída deve conter só o texto transcrito, sem comentários.

Instrução para cada página:

> {prompt}

Páginas ({total}):

{lista}
"""


def paginas() -> list[tuple[str, str]]:
    with open(MANIFESTO, encoding="utf-8") as f:
        return [(linha["obra_slug"], linha["pagina"]) for linha in csv.DictReader(f)]


def preparar():
    if PASTA_ISOLADA.exists():
        sys.exit(f"{PASTA_ISOLADA} já existe; apague-a antes de preparar de novo.")
    lista = []
    for obra, pagina in paginas():
        destino = PASTA_ISOLADA / "imagens" / obra / f"{pagina}.png"
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(IMAGENS / obra / f"{pagina}.png", destino)
        (PASTA_ISOLADA / "saidas" / obra).mkdir(parents=True, exist_ok=True)
        lista.append(f"- imagens/{obra}/{pagina}.png -> saidas/{obra}/{pagina}.txt")
    (PASTA_ISOLADA / "INSTRUCOES.md").write_text(
        INSTRUCOES.format(prompt=VLMOCR.PROMPT, total=len(lista), lista="\n".join(lista)),
        encoding="utf-8",
    )
    print(f"Pasta pronta: {PASTA_ISOLADA} ({len(lista)} imagens)")


def coletar():
    faltando = []
    for obra, pagina in paginas():
        origem = PASTA_ISOLADA / "saidas" / obra / f"{pagina}.txt"
        if not origem.exists() or not origem.read_text(encoding="utf-8").strip():
            faltando.append(f"{obra}/{pagina}")
            continue
        destino = SAIDA_BENCHMARK / obra / f"{pagina}.txt"
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origem, destino)
    total = len(paginas())
    print(f"Coletadas {total - len(faltando)}/{total} páginas em {SAIDA_BENCHMARK}")
    if faltando:
        print("Faltando:", ", ".join(faltando))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("etapa", choices=["preparar", "coletar"])
    {"preparar": preparar, "coletar": coletar}[parser.parse_args().etapa]()
