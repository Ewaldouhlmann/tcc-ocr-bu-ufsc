"""Etapa 1 do benchmark: roda os modelos sobre as páginas e salva as saídas.

Não precisa da regra ouro: o CER/WER é calculado depois, por avaliar.py, a
partir dos textos salvos aqui. Assim os modelos rodam uma única vez e a
avaliação pode ser refeita quantas vezes for preciso.

Robustez para rodar sem supervisão:
- cada modelo roda em um processo separado (_extrair_modelo.py); se um for
  encerrado por falta de memória, os demais seguem;
- cada página é salva ao terminar; rodar de novo retoma de onde parou;
- se o processo de um modelo morre, ele é relançado (até MAX_TENTATIVAS vezes)
  e a página em que morreu conta como falha; após 2 falhas a página é pulada;
- páginas presas por mais de TIMEOUT_PAGINA_S são interrompidas;
- modelos que não carregam são pulados e listados como pendentes.

Saídas em --saida (padrão data/results/extracao/):
- {modelo}/{obra}/{pagina}.txt    texto reconhecido
- metricas.csv                     tempo, CPU e RAM por página (uma linha por página)
- _estado/falhas.jsonl             falhas registradas
- _estado/resumo.txt               situação final de cada modelo
- extracao.log                     registro da execução (quando rodado com nohup)

Uso:
    python benchmark/extrair.py                      # todos os modelos mantidos
    python benchmark/extrair.py --models qwen        # só um modelo (retoma)
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ / "eda"))
from perfil_recursos import Amostrador  # noqa: E402

# Modelos mantidos após as exclusões (Seção "Modelos Excluídos antes dos
# Experimentos" do TCC), dos mais rápidos para os mais lentos: se algo der
# errado no fim, os resultados rápidos já estarão salvos.
MODELOS_PADRAO = ["tesseract", "paddleocr", "doctr", "easyocr", "llamaparse-free", "glm-ocr"]
MODELOS_OLLAMA = {"qwen", "glm-ocr", "qwen3-vl", "qwen-7b", "gemma4-e4b", "deepseek-ocr"}
MAX_TENTATIVAS = 3
TIMEOUT_PAGINA_S = 40 * 60
INTERVALO_S = 0.5
CAMPOS = ["modelo", "obra", "pagina", "tempo_s", "ram_pico_mb", "cpu_media_pct",
          "caracteres", "tokens", "concluido_em"]


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def copiar_abbyy(saida: Path):
    """O texto do ABBYY já vem nos PDFs; é só trazê-lo para a mesma estrutura."""
    origem = RAIZ / "data" / "results" / "abbyy_embutido"
    if origem.exists():
        for txt in origem.glob("*/*.txt"):
            destino = saida / "abbyy" / txt.parent.name / txt.name
            destino.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(txt, destino)


def registrar_falha(estado: Path, modelo: str, pagina: str | None, erro: str):
    with open(estado / "falhas.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"modelo": modelo, "pagina": pagina, "erro": erro[:400],
                            "quando": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False) + "\n")


def executar_tentativa(modelo: str, imagens: Path, saida: Path, metricas_csv: Path) -> str:
    """Roda o processo filho uma vez. Retorna 'concluido', 'erro_carga' ou 'morreu'."""
    estado = saida / "_estado"
    cmd = [sys.executable, str(AQUI / "_extrair_modelo.py"), modelo, str(imagens), str(saida)]
    proc = subprocess.Popen(cmd, cwd=RAIZ, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    amostrador = Amostrador(psutil.Process(proc.pid))
    atual = {"pagina": None, "inicio": None, "amostras": [], "erro_carga": False}
    trava = threading.Lock()

    def ler_eventos():
        for linha in proc.stdout:
            try:
                ev = json.loads(linha)
            except json.JSONDecodeError:
                continue
            with trava:
                if ev["evento"] == "pendentes":
                    log(f"  {modelo}: {ev['n']} página(s) pendente(s)")
                elif ev["evento"] == "erro_carga":
                    atual["erro_carga"] = True
                    log(f"  {modelo}: falha ao carregar ({ev['erro'][:120]})")
                elif ev["evento"] == "aguardando_ollama":
                    log(f"  {modelo} {ev['pagina']}: Ollama fora do ar, aguardando reinício")
                elif ev["evento"] == "aguardando_api":
                    log(f"  {modelo} {ev['pagina']}: limite da API, nova tentativa em {ev['espera_s']}s")
                    # A espera não é tempo de processamento: reinicia o cronômetro.
                    atual.update(inicio=ev["t"] + ev["espera_s"], amostras=[])
                elif ev["evento"] == "inicio_pagina":
                    atual.update(pagina=ev["pagina"], inicio=ev["t"], amostras=[])
                elif ev["evento"] == "erro_pagina":
                    log(f"  {modelo} {ev['pagina']}: ERRO {ev['erro'][:120]}")
                    atual.update(pagina=None, inicio=None)
                elif ev["evento"] == "fim_pagina":
                    a = atual["amostras"] or [{"ram": 0, "cpu": 0}]
                    obra, pagina = ev["pagina"].split("/")
                    linha_csv = {
                        "modelo": modelo, "obra": obra, "pagina": pagina,
                        "tempo_s": round(ev["t"] - atual["inicio"], 1),
                        "ram_pico_mb": round(max(x["ram"] for x in a)),
                        "cpu_media_pct": round(sum(x["cpu"] for x in a) / len(a), 1),
                        "caracteres": ev["caracteres"], "tokens": ev.get("tokens"),
                        "concluido_em": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    novo = not metricas_csv.exists()
                    with open(metricas_csv, "a", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=CAMPOS)
                        if novo:
                            w.writeheader()
                        w.writerow(linha_csv)
                    log(f"  {modelo} {ev['pagina']}: ok em {linha_csv['tempo_s']}s, "
                        f"pico {linha_csv['ram_pico_mb']} MB")
                    atual.update(pagina=None, inicio=None)

    leitor = threading.Thread(target=ler_eventos, daemon=True)
    leitor.start()

    while proc.poll() is None:
        s = amostrador.amostrar()
        usa_ollama = modelo in MODELOS_OLLAMA
        with trava:
            if atual["inicio"] is not None:
                atual["amostras"].append({
                    "ram": s["ram_python_mb"] + (s["ram_ollama_mb"] if usa_ollama else 0),
                    "cpu": s["cpu_python_pct"] + (s["cpu_ollama_pct"] if usa_ollama else 0),
                })
                if time.time() - atual["inicio"] > TIMEOUT_PAGINA_S:
                    log(f"  {modelo} {atual['pagina']}: timeout de {TIMEOUT_PAGINA_S // 60} min, interrompendo")
                    registrar_falha(estado, modelo, atual["pagina"], f"timeout ({TIMEOUT_PAGINA_S}s)")
                    (estado / f"{modelo}.atual").unlink(missing_ok=True)
                    proc.kill()
        time.sleep(INTERVALO_S)
    leitor.join(timeout=10)

    if atual["erro_carga"]:
        return "erro_carga"
    if proc.returncode == 4:
        log(f"  {modelo}: cota da API esgotada; modelo interrompido (retomar depois)")
        return "erro_carga"
    if proc.returncode == 0:
        return "concluido"
    # Morreu no meio (ex.: OOM killer): contabiliza a página em andamento.
    marcador = estado / f"{modelo}.atual"
    pagina = marcador.read_text(encoding="utf-8") if marcador.exists() else None
    marcador.unlink(missing_ok=True)
    motivo = f"processo encerrado com código {proc.returncode}"
    if proc.returncode == -9:
        motivo += " (provável falta de memória)"
    registrar_falha(estado, modelo, pagina, motivo)
    log(f"  {modelo} {pagina}: {motivo}")
    return "morreu"


def situacao(modelo: str, imagens: Path, saida: Path) -> tuple[int, int]:
    total = len(list(imagens.glob("*/*.png")))
    feitas = len(list((saida / modelo).glob("*/*.txt"))) if (saida / modelo).exists() else 0
    return feitas, total


def main():
    parser = argparse.ArgumentParser(description="Extração OCR (etapa 1 do benchmark).")
    parser.add_argument("--models", nargs="+", default=MODELOS_PADRAO)
    parser.add_argument("--images", default=str(RAIZ / "data" / "corpus" / "images"))
    parser.add_argument("--saida", default=str(RAIZ / "data" / "results" / "extracao"))
    args = parser.parse_args()

    imagens, saida = Path(args.images), Path(args.saida)
    estado = saida / "_estado"
    estado.mkdir(parents=True, exist_ok=True)
    metricas_csv = saida / "metricas.csv"
    copiar_abbyy(saida)

    log(f"Início: {len(list(imagens.glob('*/*.png')))} páginas, modelos {args.models}")
    for modelo in args.models:
        log(f"== {modelo}")
        for tentativa in range(1, MAX_TENTATIVAS + 1):
            resultado = executar_tentativa(modelo, imagens, saida, metricas_csv)
            if resultado in ("concluido", "erro_carga"):
                break
            log(f"  {modelo}: relançando (tentativa {tentativa + 1}/{MAX_TENTATIVAS})")
        feitas, total = situacao(modelo, imagens, saida)
        log(f"  {modelo}: {feitas}/{total} páginas salvas")

    linhas = ["Situação da extração — " + time.strftime("%Y-%m-%d %H:%M:%S"), ""]
    # Lista todos os modelos com saída (não só os desta execução), pois duas
    # execuções em paralelo escrevem o mesmo resumo.
    com_saida = sorted(d.name for d in saida.iterdir() if d.is_dir() and not d.name.startswith("_"))
    for modelo in dict.fromkeys(args.models + com_saida):
        feitas, total = situacao(modelo, imagens, saida)
        marca = "OK" if feitas == total else "PENDENTE"
        linhas.append(f"{marca:9} {modelo:18} {feitas:3}/{total} páginas")
    linhas += ["", "Falhas: ver _estado/falhas.jsonl. Para retomar só o que falta:",
               "  python benchmark/extrair.py --models <modelo>"]
    (estado / "resumo.txt").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    log("Fim.\n" + "\n".join(linhas))


if __name__ == "__main__":
    main()
