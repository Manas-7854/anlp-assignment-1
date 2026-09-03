"""Experiment logging, checkpoint, metric, timing, and plotting utilities."""

from __future__ import annotations

import time
import math
from collections import Counter
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
    try:
        raw_bytes = text.encode("latin-1")
    except UnicodeEncodeError:
        raw_bytes = text.encode("utf-8")
    return "".join(f"{byte:08b}" for byte in raw_bytes)


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


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def _ngrams(tokens: list[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(
        tuple(tokens[index : index + order])
        for index in range(len(tokens) - order + 1)
    )


def _corpus_bleu(predictions: Sequence[str], targets: Sequence[str]) -> float:
    """Word-level corpus BLEU-4 with add-one smoothing."""

    clipped = [0] * 4
    totals = [0] * 4
    prediction_length = target_length = 0
    for prediction, target in zip(predictions, targets):
        hypothesis = prediction.split()
        reference = target.split()
        prediction_length += len(hypothesis)
        target_length += len(reference)
        for order in range(1, 5):
            hypothesis_counts = _ngrams(hypothesis, order)
            reference_counts = _ngrams(reference, order)
            clipped[order - 1] += sum(
                min(count, reference_counts[ngram])
                for ngram, count in hypothesis_counts.items()
            )
            totals[order - 1] += sum(hypothesis_counts.values())
    if prediction_length == 0:
        return 0.0
    precisions = [
        (matches + 1) / (total + 1)
        for matches, total in zip(clipped, totals)
    ]
    brevity_penalty = (
        1.0
        if prediction_length > target_length
        else math.exp(1.0 - target_length / prediction_length)
    )
    return brevity_penalty * math.exp(
        sum(math.log(precision) for precision in precisions) / 4
    )


def _f1(overlap: int, predicted: int, reference: int) -> float:
    if predicted == 0 or reference == 0 or overlap == 0:
        return 0.0
    precision = overlap / predicted
    recall = overlap / reference
    return 2 * precision * recall / (precision + recall)


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1
                if left_item == right_item
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def _rouge_scores(predictions: Sequence[str], targets: Sequence[str]) -> dict[str, float]:
    totals = {"rouge1_f1": 0.0, "rouge2_f1": 0.0, "rougeL_f1": 0.0}
    if not targets:
        return totals
    for prediction, target in zip(predictions, targets):
        hypothesis = prediction.split()
        reference = target.split()
        for order, name in ((1, "rouge1_f1"), (2, "rouge2_f1")):
            hypothesis_counts = _ngrams(hypothesis, order)
            reference_counts = _ngrams(reference, order)
            overlap = sum(
                min(count, reference_counts[ngram])
                for ngram, count in hypothesis_counts.items()
            )
            totals[name] += _f1(
                overlap,
                sum(hypothesis_counts.values()),
                sum(reference_counts.values()),
            )
        totals["rougeL_f1"] += _f1(
            _lcs_length(hypothesis, reference), len(hypothesis), len(reference)
        )
    return {name: value / len(targets) for name, value in totals.items()}


def compute_evaluation_metrics(
    predictions: Sequence[str],
    targets: Sequence[str],
    include_tokenized_metrics: bool = True,
) -> dict[str, float]:
    """Compute required greedy-decoding metrics for one test set."""

    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have equal lengths.")
    distances = [
        _edit_distance(prediction, target)
        for prediction, target in zip(predictions, targets)
    ]
    metrics = {
        "bit_level_accuracy": bit_level_accuracy(predictions, targets),
        "sequence_accuracy": sequence_accuracy(predictions, targets),
        "levenshtein_distance": sum(distances) / len(distances) if distances else 0.0,
    }
    if not include_tokenized_metrics:
        return metrics

    metrics["bleu"] = _corpus_bleu(predictions, targets)
    metrics.update(_rouge_scores(predictions, targets))
    return metrics


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
