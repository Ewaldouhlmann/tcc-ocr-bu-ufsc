"""Processo filho usado por perfil_recursos.py: carrega UM modelo e processa
as páginas indicadas, emitindo marcadores JSON (uma linha por evento) em
stdout para que o processo pai saiba em que fase o modelo está.

Rodar cada modelo em um processo separado isola a memória (o pico medido é só
daquele modelo) e evita que uma falha por falta de memória derrube os demais.
"""

import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))


def emitir(evento: str, **dados):
    print(json.dumps({"evento": evento, "t": time.time(), **dados}), flush=True)


def main():
    nome = sys.argv[1]
    saida = Path(sys.argv[2])
    paginas = [Path(p) for p in sys.argv[3:]]

    emitir("inicio_carga")
    try:
        from src.ocr import MODEL_REGISTRY

        modelo = MODEL_REGISTRY[nome]()
    except Exception as e:  # noqa: BLE001 - qualquer falha de carga é registrada
        emitir("erro", fase="carga", erro=f"{type(e).__name__}: {str(e)[:300]}")
        return
    emitir("fim_carga")

    for pagina in paginas:
        emitir("inicio_pagina", pagina=pagina.stem)
        try:
            texto = modelo.recognize(pagina.resolve())
        except Exception as e:  # noqa: BLE001
            emitir("erro", fase="inferencia", pagina=pagina.stem,
                   erro=f"{type(e).__name__}: {str(e)[:300]}")
            continue
        destino = saida / nome
        destino.mkdir(parents=True, exist_ok=True)
        (destino / f"{pagina.stem}.txt").write_text(texto, encoding="utf-8")
        emitir("fim_pagina", pagina=pagina.stem,
               tokens=getattr(modelo, "last_tokens_used", None), caracteres=len(texto))


if __name__ == "__main__":
    main()
