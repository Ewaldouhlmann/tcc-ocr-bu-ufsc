"""Perfil de uso de recursos (tempo, CPU e RAM) de cada modelo OCR.

Para cada modelo, um processo filho (_executar_modelo.py) carrega o modelo e
processa as páginas, enquanto este processo amostra, a cada INTERVALO_S:

- CPU e RAM (RSS) da árvore de processos do filho;
- CPU e RAM dos processos do Ollama (servidor + runner), pois nos VLMs locais
  a inferência não acontece no processo Python, e sim no Ollama. Medir só o
  processo Python (como faz resource.getrusage no run_benchmark.py)
  subestimaria a memória dos VLMs em várias ordens de grandeza.

CPU é expressa em % da máquina inteira (100% = todos os núcleos ocupados).
Para modelos via API, os valores medidos são apenas do cliente local; o custo
computacional real fica no provedor.

Saídas (em eda/dados/):
- perfil_linha_tempo.csv : uma linha por amostra (modelo, t, fase, cpu, ram)
- perfil_eventos.csv     : marcadores de fase e erros emitidos pelo filho
- saidas/{modelo}/{pagina}.txt : texto reconhecido

Uso:
    python eda/perfil_recursos.py                       # todos os modelos padrão
    python eda/perfil_recursos.py --models tesseract qwen
"""

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
DADOS = AQUI / "dados"
INTERVALO_S = 0.25
TIMEOUT_S = 90 * 60  # VLMs locais em CPU levam ~9 min por página

# TrOCR fica de fora (excluído por requisito de GPU, Seção 5 do TCC) e o
# LlamaParse Agentic também (eliminado por custo na triagem).
MODELOS_PADRAO = [
    "tesseract", "easyocr", "paddleocr", "doctr",
    "qwen", "qwen3-vl", "glm-ocr",
    "llamaparse-free", "mistral", "gemini",
]

# Tags usadas pelos adaptadores Ollama, para descarregar o modelo ao final.
TAGS_OLLAMA = {"qwen": "qwen2.5vl:3b", "qwen3-vl": "qwen3-vl:2b", "glm-ocr": "glm-ocr:q8_0",
               "gemma4": "gemma4:2b", "deepseek": "deepseek-vl2:1b"}


def _processos_ollama() -> list[psutil.Process]:
    procs = []
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            nome = (p.info["name"] or "").lower()
            cmd = " ".join(p.info["cmdline"] or []).lower()
            if "ollama" in nome or "ollama" in cmd:
                procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return procs


class Amostrador:
    """Amostra CPU/RAM de um processo (com filhos) e dos processos do Ollama."""

    def __init__(self, raiz: psutil.Process):
        self.raiz = raiz
        self.cache: dict[int, psutil.Process] = {}
        self.ncpu = psutil.cpu_count() or 1

    def _proc(self, p: psutil.Process) -> psutil.Process:
        # cpu_percent() precisa do MESMO objeto entre chamadas para calcular o delta.
        if p.pid not in self.cache:
            self.cache[p.pid] = p
            p.cpu_percent(None)
        return self.cache[p.pid]

    def amostrar(self) -> dict:
        grupos = {"python": [], "ollama": []}
        try:
            grupos["python"] = [self.raiz] + self.raiz.children(recursive=True)
        except psutil.NoSuchProcess:
            pass
        grupos["ollama"] = _processos_ollama()
        linha = {}
        for grupo, procs in grupos.items():
            cpu = ram = 0.0
            for p in procs:
                try:
                    p = self._proc(p)
                    cpu += p.cpu_percent(None)
                    ram += p.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            linha[f"cpu_{grupo}_pct"] = round(cpu / self.ncpu, 1)
            linha[f"ram_{grupo}_mb"] = round(ram / 1024**2, 1)
        vm = psutil.virtual_memory()
        linha["ram_sistema_usada_mb"] = round((vm.total - vm.available) / 1024**2, 1)
        return linha


def perfilar(modelo: str, paginas: list[Path], saida: Path) -> tuple[list[dict], list[dict]]:
    cmd = [sys.executable, str(AQUI / "_executar_modelo.py"), modelo, str(saida)] + [str(p) for p in paginas]
    proc = subprocess.Popen(cmd, cwd=RAIZ, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    eventos: list[dict] = []
    fase = {"atual": "inicializacao", "pagina": ""}

    def ler_eventos():
        for linha in proc.stdout:
            try:
                ev = json.loads(linha)
            except json.JSONDecodeError:
                continue  # prints de bibliotecas
            eventos.append(ev)
            if ev["evento"] == "inicio_carga":
                fase["atual"] = "carga"
            elif ev["evento"] == "inicio_pagina":
                fase.update(atual="inferencia", pagina=ev["pagina"])
            elif ev["evento"] in ("fim_pagina", "fim_carga"):
                fase.update(atual="ocioso", pagina="")

    leitor = threading.Thread(target=ler_eventos, daemon=True)
    leitor.start()

    amostrador = Amostrador(psutil.Process(proc.pid))
    amostras = []
    inicio = time.time()
    while proc.poll() is None:
        if time.time() - inicio > TIMEOUT_S:
            proc.kill()
            eventos.append({"evento": "erro", "t": time.time(), "fase": fase["atual"],
                            "erro": f"timeout ({TIMEOUT_S}s)"})
            break
        amostra = amostrador.amostrar()
        amostras.append({"modelo": modelo, "t_s": round(time.time() - inicio, 2),
                         "fase": fase["atual"], "pagina": fase["pagina"], **amostra})
        time.sleep(INTERVALO_S)
    leitor.join(timeout=5)

    if proc.returncode not in (0, None) and not any(e["evento"] == "erro" for e in eventos):
        # Morto por sinal (ex.: OOM killer) sem conseguir registrar o erro.
        eventos.append({"evento": "erro", "t": time.time(), "fase": fase["atual"],
                        "erro": f"processo encerrado com código {proc.returncode}"})

    if modelo in TAGS_OLLAMA:
        subprocess.run(["ollama", "stop", TAGS_OLLAMA[modelo]], capture_output=True)

    for ev in eventos:
        ev["modelo"] = modelo
        ev["t_s"] = round(ev.pop("t") - inicio, 2)
    return amostras, eventos


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=MODELOS_PADRAO)
    parser.add_argument("--paginas", default=str(DADOS / "paginas" / "sintetica"),
                        help="Pasta com as páginas PNG")
    parser.add_argument("--dados", default=str(DADOS),
                        help="Pasta onde salvar CSVs e textos (padrão: eda/dados)")
    args = parser.parse_args()
    dados = Path(args.dados)

    paginas = sorted(Path(args.paginas).glob("*.png"))
    if not paginas:
        sys.exit(f"Nenhuma página em {args.paginas}; rode eda/pagina_sintetica.py antes.")
    dados.mkdir(parents=True, exist_ok=True)
    saida = dados / "saidas"

    linha_tempo_csv = dados / "perfil_linha_tempo.csv"
    eventos_csv = dados / "perfil_eventos.csv"
    # Reexecuções substituem apenas os modelos pedidos.
    antigas_lt, antigos_ev = [], []
    for caminho, destino in ((linha_tempo_csv, antigas_lt), (eventos_csv, antigos_ev)):
        if caminho.exists():
            with open(caminho, encoding="utf-8") as f:
                destino.extend(r for r in csv.DictReader(f) if r["modelo"] not in args.models)

    todas_amostras, todos_eventos = antigas_lt, antigos_ev
    for modelo in args.models:
        print(f"[{modelo}] perfilando {len(paginas)} página(s)...", flush=True)
        amostras, eventos = perfilar(modelo, paginas, saida)
        erros = [e for e in eventos if e["evento"] == "erro"]
        duracao = amostras[-1]["t_s"] if amostras else 0
        usa_ollama = modelo in TAGS_OLLAMA
        pico = max((a["ram_python_mb"] + (a["ram_ollama_mb"] if usa_ollama else 0) for a in amostras), default=0)
        status = "ERRO: " + erros[0]["erro"] if erros else "ok"
        print(f"[{modelo}] {duracao:.0f}s, pico RAM {pico:.0f} MB, {status}", flush=True)
        todas_amostras += amostras
        todos_eventos += eventos

    campos_ev = ["modelo", "evento", "t_s", "fase", "pagina", "tokens", "caracteres", "erro"]
    for caminho, linhas, campos in (
        (linha_tempo_csv, todas_amostras, None),
        (eventos_csv, todos_eventos, campos_ev),
    ):
        campos = campos or list(linhas[0].keys())
        with open(caminho, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
            w.writeheader()
            w.writerows(linhas)
    print(f"Dados salvos em {dados}")


if __name__ == "__main__":
    main()
