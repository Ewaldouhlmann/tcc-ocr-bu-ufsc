"""Cálculo de WER, CER e geração de relatórios comparativos."""

from jiwer import wer, cer
from pathlib import Path


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
    Retorna um dicionário com WER e CER médios e por página."""
    results = {"model": model_name, "pages": [], "wer_mean": None, "cer_mean": None}
    # TODO: implementar iteração sobre páginas
    return results
