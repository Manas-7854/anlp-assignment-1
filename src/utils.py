"""Experiment logging, checkpoint, metric, timing, and plotting utilities."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import torch


def initialize_wandb(
    config_name: str,
    config: dict[str, Any],
    project: str = "anlp-assignment-1",
    mode: str = "online",
):
    try:
        import wandb
    except ImportError as error:
        if mode == "disabled":
            return None
        raise RuntimeError("Install wandb or use --wandb-mode disabled.") from error

    return wandb.init(project=project, name=config_name, config=config, mode=mode)


def log_training_step(
    run, global_step: int, loss: float, learning_rate: float
) -> None:
    if run is not None:
        run.log(
            {
                "global_step": global_step,
                "train/step_loss": loss,
                "learning_rate": learning_rate,
            },
            step=global_step,
            commit=False,
        )


def log_epoch(
    run,
    global_step: int,
    epoch: int,
    train_loss: float,
    validation_loss: float,
    validation_token_accuracy: float,
    learning_rate: float,
    epoch_seconds: float,
    total_seconds: float,
    tokens_per_second: float,
    peak_memory_mb: float,
) -> None:
    if run is not None:
        run.log(
            {
                "global_step": global_step,
                "epoch": epoch,
                "train/loss": train_loss,
                "validation/loss": validation_loss,
                "validation/token_accuracy": validation_token_accuracy,
                "learning_rate": learning_rate,
                "time/epoch_seconds": epoch_seconds,
                "time/total_seconds": total_seconds,
                "performance/tokens_per_second": tokens_per_second,
                "gpu/peak_memory_mb": peak_memory_mb,
            },
            step=global_step,
        )


def log_final_evaluation(run, metrics: dict[str, float]) -> None:
    if run is None:
        return
    payload = {
        f"test/{name}": value
        for name, value in metrics.items()
        if name not in {"training_time_seconds", "gpu_peak_memory_mb"}
    }
    payload["time/total_training_seconds"] = metrics["training_time_seconds"]
    payload["gpu/peak_memory_mb"] = metrics["gpu_peak_memory_mb"]
    run.log(payload)


def start_timer() -> float:
    return time.perf_counter()


def elapsed_seconds(start_time: float) -> float:
    return time.perf_counter() - start_time


def reset_gpu_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def gpu_peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def _as_bits(text: str) -> str:
    if set(text) <= {"0", "1"}:
        return text
    return "".join(f"{byte:08b}" for byte in text.encode("utf-8"))


def bit_level_accuracy(predictions: Sequence[str], targets: Sequence[str]) -> float:
    correct = total = 0
    for prediction, target in zip(predictions, targets):
        predicted_bits, target_bits = _as_bits(prediction), _as_bits(target)
        correct += sum(a == b for a, b in zip(predicted_bits, target_bits))
        total += max(len(predicted_bits), len(target_bits))
    return correct / total if total else 1.0


def sequence_accuracy(predictions: Sequence[str], targets: Sequence[str]) -> float:
    if not targets:
        return 0.0
    return sum(a == b for a, b in zip(predictions, targets)) / len(targets)


def compute_evaluation_metrics(
    predictions: Sequence[str], targets: Sequence[str]
) -> dict[str, float]:
    """Compute text metrics using NLTK and rouge-score implementations."""

    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have equal lengths.")
    try:
        from nltk.metrics.distance import edit_distance
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
        from rouge_score.rouge_scorer import RougeScorer
    except ImportError as error:
        raise RuntimeError(
            "Final evaluation requires nltk and rouge-score."
        ) from error

    distances = [
        edit_distance(prediction, target)
        for prediction, target in zip(predictions, targets)
    ]
    references = [[target.split()] for target in targets]
    hypotheses = [prediction.split() for prediction in predictions]
    bleu = (
        corpus_bleu(
            references,
            hypotheses,
            smoothing_function=SmoothingFunction().method1,
        )
        if hypotheses
        else 0.0
    )

    scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge = [
        scorer.score(target, prediction)
        for prediction, target in zip(predictions, targets)
    ]
    count = len(rouge)
    return {
        "bit_level_accuracy": bit_level_accuracy(predictions, targets),
        "sequence_accuracy": sequence_accuracy(predictions, targets),
        "levenshtein_distance": sum(distances) / len(distances) if distances else 0.0,
        "bleu": bleu,
        "rouge1_f1": (
            sum(score["rouge1"].fmeasure for score in rouge) / count
            if count
            else 0.0
        ),
        "rouge2_f1": (
            sum(score["rouge2"].fmeasure for score in rouge) / count
            if count
            else 0.0
        ),
        "rougeL_f1": (
            sum(score["rougeL"].fmeasure for score in rouge) / count
            if count
            else 0.0
        ),
    }


def plot_training_history(
    history: dict[str, Sequence[float]], path: str | Path
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axis = plt.subplots()
    axis.plot(epochs, history["train_loss"], label="train")
    axis.plot(epochs, history["validation_loss"], label="validation")
    axis.set(xlabel="Epoch", ylabel="Loss", title="Training losses")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path
