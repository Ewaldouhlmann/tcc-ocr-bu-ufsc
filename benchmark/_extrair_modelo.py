"""Processo filho de extrair.py: carrega UM modelo e transcreve as páginas que
ainda não têm saída salva.

Cada página é gravada assim que termina, então uma interrupção (falta de
memória, timeout, queda de energia) perde no máximo a página em andamento.
Páginas que já falharam 2 vezes são puladas, para não travar o modelo inteiro
em uma página problemática.

Emite eventos JSON (uma linha por evento) em stdout para o processo pai medir
tempo, CPU e RAM por página.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

MAX_FALHAS_POR_PAGINA = 2
# Preenchido após carregar o modelo: adaptadores Ollama têm model_tag.
TAGS_OLLAMA: dict[str, str] = {}


def emitir(evento: str, **dados):
    print(json.dumps({"evento": evento, "t": time.time(), **dados}), flush=True)


def contar_falhas(estado: Path, modelo: str) -> dict[str, int]:
    contagem: dict[str, int] = {}
    arquivo = estado / "falhas.jsonl"
    if arquivo.exists():
        for linha in arquivo.read_text(encoding="utf-8").splitlines():
            f = json.loads(linha)
            if f["modelo"] == modelo and f.get("pagina"):
                contagem[f["pagina"]] = contagem.get(f["pagina"], 0) + 1
    return contagem


def registrar_falha(estado: Path, modelo: str, pagina: str | None, erro: str):
    with open(estado / "falhas.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"modelo": modelo, "pagina": pagina, "erro": erro[:400],
                            "quando": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False) + "\n")


def ollama_no_ar(limite_s: int = 300) -> bool:
    """Espera o servidor do Ollama responder (o systemd o reinicia em ~3 s
    depois de uma queda por falta de memória)."""
    import urllib.request

    fim = time.time() + limite_s
    while time.time() < fim:
        try:
            urllib.request.urlopen("http://localhost:11434/api/version", timeout=3)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(3)
    return False


def _sem_conexao_ollama(e: Exception) -> bool:
    return "Failed to connect to Ollama" in str(e) or type(e).__name__ == "ConnectError"


def _ollama_caiu(e: Exception) -> bool:
    return "Server disconnected" in str(e) or "unexpected EOF" in str(e)


def _limite_api(e: Exception) -> bool:
    msg = str(e)
    return ("429" in msg or type(e).__name__ == "ResourceExhausted"
            or "quota" in msg.lower() or "rate limit" in msg.lower())


def reconhecer(modelo, modelo_nome: str, png: Path, estado: Path, chave: str) -> str | None:
    """Reconhece uma página tratando falhas transitórias. Devolve None se falhou."""
    esperas_api = [60, 120, 240, 480]
    tentativas_conexao = 0
    while True:
        try:
            return modelo.recognize(png.resolve())
        except Exception as e:  # noqa: BLE001
            erro = f"{type(e).__name__}: {e}"
            if modelo_nome in TAGS_OLLAMA and _sem_conexao_ollama(e) and tentativas_conexao < 5:
                # Ollama fora do ar (reiniciando): não é culpa desta página.
                tentativas_conexao += 1
                emitir("aguardando_ollama", pagina=chave)
                ollama_no_ar()
                continue
            if modelo_nome in TAGS_OLLAMA and _ollama_caiu(e):
                # Esta página derrubou o servidor (em geral, falta de memória).
                registrar_falha(estado, modelo_nome, chave, erro)
                emitir("erro_pagina", pagina=chave, erro=erro[:300])
                emitir("aguardando_ollama", pagina=chave)
                ollama_no_ar()
                return None
            if _limite_api(e) and esperas_api:
                espera = esperas_api.pop(0)
                emitir("aguardando_api", pagina=chave, espera_s=espera)
                time.sleep(espera)
                continue
            if _limite_api(e):
                # Limite persistente (cota esgotada): insistir nas demais páginas
                # só gastaria horas de espera. Encerra o modelo; retoma depois.
                registrar_falha(estado, modelo_nome, chave, erro)
                emitir("erro_pagina", pagina=chave, erro=erro[:300])
                emitir("cota_esgotada")
                sys.exit(4)
            registrar_falha(estado, modelo_nome, chave, erro)
            emitir("erro_pagina", pagina=chave, erro=erro[:300])
            return None


def main():
    modelo_nome, imagens, saida = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    estado = saida / "_estado"
    estado.mkdir(parents=True, exist_ok=True)

    falhas = contar_falhas(estado, modelo_nome)
    pendentes = []
    for png in sorted(imagens.glob("*/*.png")):
        chave = f"{png.parent.name}/{png.stem}"
        if (saida / modelo_nome / png.parent.name / f"{png.stem}.txt").exists():
            continue
        if falhas.get(chave, 0) >= MAX_FALHAS_POR_PAGINA:
            continue
        pendentes.append((chave, png))
    emitir("pendentes", n=len(pendentes))
    if not pendentes:
        return

    emitir("inicio_carga")
    try:
        from src.ocr import MODEL_REGISTRY

        modelo = MODEL_REGISTRY[modelo_nome]()
    except Exception as e:  # noqa: BLE001
        registrar_falha(estado, modelo_nome, None, f"carga: {type(e).__name__}: {e}")
        emitir("erro_carga", erro=f"{type(e).__name__}: {str(e)[:300]}")
        sys.exit(3)
    from src.ocr.vlm import OllamaVLM

    if isinstance(modelo, OllamaVLM):
        TAGS_OLLAMA[modelo_nome] = modelo.model_tag
    emitir("fim_carga")

    for chave, png in pendentes:
        # Marca a página em andamento: se o processo morrer (ex.: OOM killer),
        # o pai sabe qual página contabilizar como falha.
        (estado / f"{modelo_nome}.atual").write_text(chave, encoding="utf-8")
        emitir("inicio_pagina", pagina=chave)
        texto = reconhecer(modelo, modelo_nome, png, estado, chave)
        if modelo_nome in TAGS_OLLAMA:
            # Descarrega o modelo entre páginas: sem isso o Qwen2.5-VL
            # acumula memória e é encerrado na página seguinte. Recarregar
            # leva ~1 s, pois os pesos ficam no cache de disco do sistema.
            subprocess.run(["ollama", "stop", TAGS_OLLAMA[modelo_nome]], capture_output=True)
        if texto is None:
            continue
        if not texto.strip():
            # Todas as páginas da amostra têm texto: saída vazia é falha do
            # modelo (ex.: Qwen3-VL com raciocínio esgotando o contexto).
            registrar_falha(estado, modelo_nome, chave, "saída vazia")
            emitir("erro_pagina", pagina=chave, erro="saída vazia")
            continue
        destino = saida / modelo_nome / png.parent.name
        destino.mkdir(parents=True, exist_ok=True)
        (destino / f"{png.stem}.txt").write_text(texto, encoding="utf-8")
        emitir("fim_pagina", pagina=chave, caracteres=len(texto),
               tokens=getattr(modelo, "last_tokens_used", None))
    (estado / f"{modelo_nome}.atual").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
