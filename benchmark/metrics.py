"""Cálculo de WER, CER e geração de relatórios comparativos."""

from pathlib import Path

from jiwer import cer, wer


def compute_wer(hypothesis: str, reference: str) -> float:
    return wer(reference, hypothesis)


def compute_cer(hypothesis: str, reference: str) -> float:
    return cer(reference, hypothesis)


def evaluate_model(
    predictions_dir: Path,
    ground_truth_dir: Path,
    model_name: str,
) -> dict:
    """Avalia um modelo comparando suas predições com o ground truth.

    Casa cada predição em predictions_dir/{pagina}.txt com a referência de
    mesmo nome em ground_truth_dir. Páginas sem referência correspondente são
    ignoradas. Retorna um dicionário com WER e CER médios e por página.
    """
    results = {"model": model_name, "pages": [], "wer_mean": None, "cer_mean": None}

    for prediction_path in sorted(Path(predictions_dir).glob("*.txt")):
        reference_path = Path(ground_truth_dir) / prediction_path.name
        if not reference_path.exists():
            print(f"  [{model_name}] sem ground truth para {prediction_path.name}, ignorando")
            continue

        hypothesis = prediction_path.read_text(encoding="utf-8")
        reference = reference_path.read_text(encoding="utf-8")

        results["pages"].append({
            "page": prediction_path.stem,
            "wer": compute_wer(hypothesis, reference),
            "cer": compute_cer(hypothesis, reference),
        })

    if results["pages"]:
        n = len(results["pages"])
        results["wer_mean"] = sum(p["wer"] for p in results["pages"]) / n
        results["cer_mean"] = sum(p["cer"] for p in results["pages"]) / n

    return results


#: Colunas da aba "Resultados", na ordem em que aparecem na planilha.
REPORT_COLUMNS = [
    ("modelo", "Modelo"),
    ("obra", "Obra"),
    ("pagina", "Página"),
    ("dificuldade", "Classificação"),
    ("cer", "CER"),
    ("wer", "WER"),
    ("tempo_s", "Tempo (s)"),
    ("memoria_ram_mb", "Memória RAM (MB)"),
    ("memoria_gpu_mb", "Memória GPU (MB)"),
    ("tokens", "Tokens"),
]

#: Colunas numéricas da aba "Resumo por Modelo" e a coluna de origem em REPORT_COLUMNS.
_SUMMARY_METRICS = [
    ("cer", "CER médio"),
    ("wer", "WER médio"),
    ("tempo_s", "Tempo médio (s)"),
    ("memoria_ram_mb", "Memória RAM média (MB)"),
    ("memoria_gpu_mb", "Memória GPU média (MB)"),
    ("tokens", "Tokens médios"),
]


def save_report(rows: list[dict], output_path: Path) -> None:
    """Salva os resultados do benchmark em uma planilha Excel (.xlsx).

    rows é uma lista de dicionários, um por página x modelo, com as chaves de
    REPORT_COLUMNS (ex.: modelo, obra, pagina, dificuldade, cer, wer, tempo_s,
    memoria_ram_mb, memoria_gpu_mb, tokens). Valores ausentes (ex.: sem ground
    truth ou sem GPU) podem ser None.

    Gera duas abas: "Resultados" (uma linha por página x modelo) e "Resumo por
    Modelo" (médias agregadas por modelo, ignorando valores None).
    """
    from openpyxl import Workbook

    wb = Workbook()

    ws = wb.active
    ws.title = "Resultados"
    keys = [key for key, _ in REPORT_COLUMNS]
    ws.append([header for _, header in REPORT_COLUMNS])
    for row in rows:
        ws.append([row.get(key) for key in keys])

    ws_summary = wb.create_sheet("Resumo por Modelo")
    ws_summary.append(["Modelo", "Nº páginas"] + [header for _, header in _SUMMARY_METRICS])
    for model_name in sorted({row["modelo"] for row in rows}):
        model_rows = [row for row in rows if row["modelo"] == model_name]
        summary_row = [model_name, len(model_rows)]
        for key, _ in _SUMMARY_METRICS:
            values = [row[key] for row in model_rows if row.get(key) is not None]
            summary_row.append(sum(values) / len(values) if values else None)
        ws_summary.append(summary_row)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
