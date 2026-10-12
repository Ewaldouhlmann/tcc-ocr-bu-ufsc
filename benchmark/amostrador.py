"""Amostragem de CPU e RAM durante a extração (usada por extrair.py).

Nos VLMs locais a inferência acontece no servidor do Ollama, não no processo
Python; por isso, além do processo do modelo (e filhos), somam-se os processos
do Ollama. Medir só o processo Python subestimaria a memória desses modelos.
CPU em % da máquina inteira (100% = todos os núcleos ocupados).
"""

import psutil


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
