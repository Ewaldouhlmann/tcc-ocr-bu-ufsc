"""Ponto de entrada CLI para execução do benchmark comparativo."""

import argparse
import csv
import gc
import resource
import sys
import time
from pathlib import Path

# Permite rodar tanto via "python benchmark/run_benchmark.py" (README) quanto
# via "python -m benchmark.run_benchmark", garantindo que "src" e "benchmark"
# sejam importáveis como pacotes a partir da raiz do repositório.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.metrics import compute_cer, compute_wer, save_report  # noqa: E402
from src.ocr import MODEL_REGISTRY  # noqa: E402

AVAILABLE_MODELS = list(MODEL_REGISTRY.keys())

DEFAULT_MANIFEST = Path(__file__).resolve().parent / "ground_truth" / "manifest.csv"


def load_manifest(manifest_path: Path) -> dict[tuple[str, str], dict]:
    """Lê o manifesto obra/página/dificuldade (Tabela 4.4 do TCC).

    Retorna {(obra_slug, pagina): {"obra": ..., "dificuldade": ..., ...}}, com
    "pagina" no mesmo formato de 3 dígitos usado pelos arquivos gerados por
    src/pipeline/converter.py (ex.: "025").
    """
    manifest = {}
    if not manifest_path.exists():
        return manifest
    with open(manifest_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            manifest[(row["obra_slug"], row["pagina"])] = row
    return manifest


def _gpu_memory_mb() -> float | None:
    """Pico de memória de GPU alocada via CUDA desde o último reset, em MB."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**2)
    except ImportError:
        pass
    return None


def _reset_gpu_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def run_model(
    model_name: str,
    images_dir: Path,
    ground_truth_dir: Path,
    manifest: dict[tuple[str, str], dict],
    predictions_dir: Path,
) -> list[dict]:
    """Executa um modelo sobre todo o corpus e retorna uma linha de resultado
    por página, com métricas de qualidade (CER/WER) e de custo (tempo,
    memória RAM/GPU, tokens).

    images_dir deve conter uma subpasta por obra, cada uma com as páginas em
    PNG nomeadas "{pagina:03d}.png" (saída de src/pipeline/converter.py).
    ground_truth_dir deve espelhar essa estrutura com arquivos .txt.
    As predições brutas também são salvas em
    predictions_dir/{model_name}/{obra}/{pagina}.txt para inspeção qualitativa.
    """
    rows = []
    model = MODEL_REGISTRY[model_name]()
    model_predictions_dir = predictions_dir / model_name

    for obra_dir in sorted(p for p in Path(images_dir).iterdir() if p.is_dir()):
        obra_slug = obra_dir.name
        for image_path in sorted(obra_dir.glob("*.png")):
            pagina = image_path.stem
            print(f"  [{model_name}] {obra_slug}/{image_path.name}")

            gc.collect()
            _reset_gpu_memory()
            mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            start = time.perf_counter()

            hypothesis = model.recognize(image_path)

            elapsed = time.perf_counter() - start
            mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            gpu_mem = _gpu_memory_mb()

            output_dir = model_predictions_dir / obra_slug
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{pagina}.txt").write_text(hypothesis, encoding="utf-8")

            reference_path = Path(ground_truth_dir) / obra_slug / f"{pagina}.txt"
            reference = reference_path.read_text(encoding="utf-8") if reference_path.exists() else None

            info = manifest.get((obra_slug, pagina), {})

            rows.append({
                "modelo": model_name,
                "obra": info.get("obra", obra_slug),
                "pagina": pagina,
                "dificuldade": info.get("dificuldade"),
                "cer": compute_cer(hypothesis, reference) if reference is not None else None,
                "wer": compute_wer(hypothesis, reference) if reference is not None else None,
                "tempo_s": round(elapsed, 3),
                "memoria_ram_mb": round((mem_after - mem_before) / 1024, 2),
                "memoria_gpu_mb": round(gpu_mem, 2) if gpu_mem is not None else None,
                "tokens": getattr(model, "last_tokens_used", None),
            })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Benchmark comparativo de modelos OCR.")
    parser.add_argument(
        "--images", required=True,
        help="Pasta do corpus, com uma subpasta por obra (--images/{obra}/{pagina}.png)",
    )
    parser.add_argument(
        "--ground-truth", required=True,
        help="Pasta com transcrições de referência, espelhando --images (.../{obra}/{pagina}.txt)",
    )
    parser.add_argument(
        "--manifest", default=str(DEFAULT_MANIFEST),
        help=f"CSV obra/página/dificuldade (padrão: {DEFAULT_MANIFEST})",
    )
    parser.add_argument("--models", nargs="+", choices=AVAILABLE_MODELS + ["all"],
                        default=["all"], help="Modelos a avaliar")
    parser.add_argument("--output", required=True, help="Pasta para salvar resultados")
    args = parser.parse_args()

    models = AVAILABLE_MODELS if "all" in args.models else args.models
    manifest = load_manifest(Path(args.manifest))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Modelos selecionados: {models}")

    all_rows = []
    for model_name in models:
        print(f"Executando {model_name}...")
        all_rows.extend(run_model(
            model_name, Path(args.images), Path(args.ground_truth), manifest, output_dir,
        ))

    report_path = output_dir / "benchmark_resultados.xlsx"
    save_report(all_rows, report_path)
    print(f"Relatório salvo em {report_path}")


if __name__ == "__main__":
    main()
