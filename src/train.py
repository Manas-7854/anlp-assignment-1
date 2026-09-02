"""Train C1-C4 encoder-decoder Transformer configurations."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from dataset import (
    CIPHER_PATH,
    MERGE_RULES_PATH,
    PLAIN_PATH,
    BinaryBPE,
    PlaintextCodec,
    build_dataloaders,
)
from models.attention import GroupedQueryAttention, MultiHeadAttention
from models.norm import LayerNorm, RMSNorm
from models.positional import (
    RotaryPositionalEmbedding,
    SinusoidalPositionalEncoding,
)
from utils import (
    compute_evaluation_metrics,
    elapsed_seconds,
    gpu_peak_memory_mb,
    initialize_wandb,
    load_checkpoint,
    log_epoch,
    log_final_evaluation,
    log_training_step,
    plot_training_history,
    reset_gpu_peak_memory,
    save_checkpoint,
    start_timer,
)


ROOT = Path(__file__).resolve().parents[1]

BASE_CONFIG: dict[str, Any] = {
    "d_model": 256,
    "num_layers": 4,
    "num_heads": 8,
    "num_kv_heads": 2,
    "d_ff": 1024,
    "dropout": 0.1,
    "batch_size": 8,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "epochs": 20,
    "max_seq_length": 4096,
    "max_decode_length": 4096,
    "seed": 42,
    "train_ratio": 0.8,
    "validation_ratio": 0.1,
    "test_ratio": 0.1,
    "num_workers": 0,
    "log_every": 50,
    "device": "auto",
    "wandb_project": "anlp-assignment-1",
    "wandb_mode": "online",
    "evaluate_test": False,
    "plain_path": str(PLAIN_PATH),
    "cipher_path": str(CIPHER_PATH),
    "merge_rules_path": str(MERGE_RULES_PATH),
    "output_dir": str(ROOT / "outputs"),
}

ARCHITECTURE_CONFIGS = {
    "C1": {
        "attention": "mha",
        "positional": "sinusoidal",
        "norm": "layernorm",
    },
    "C2": {
        "attention": "mha",
        "positional": "rope",
        "norm": "layernorm",
    },
    "C3": {
        "attention": "gqa",
        "positional": "sinusoidal",
        "norm": "layernorm",
    },
    "C4": {
        "attention": "mha",
        "positional": "sinusoidal",
        "norm": "rmsnorm",
    },
}


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.dropout(self.activation(x))
        return self.output_projection(x)


class EncoderBlock(nn.Module):
    """Pre-LN encoder self-attention followed by a Pre-LN FFN."""

    def __init__(
        self,
        d_model: int,
        attention: nn.Module,
        norm_class: type[nn.Module],
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention = attention
        self.attention_norm = norm_class(d_model)
        self.ffn_norm = norm_class(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.attention_norm(x)
        attended, _ = self.self_attention(
            normalized,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
        )
        x = x + self.dropout(attended)
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


def _norm_class(config: dict[str, Any]) -> type[nn.Module]:
    return LayerNorm if config["norm"] == "layernorm" else RMSNorm


def _make_attention(
    config: dict[str, Any], use_rope: bool = True
) -> nn.Module:
    if config["attention"] == "gqa":
        return GroupedQueryAttention(
            config["d_model"],
            config["num_heads"],
            config["num_kv_heads"],
            dropout=config["dropout"],
        )

    rope = None
    if use_rope and config["positional"] == "rope":
        rope = RotaryPositionalEmbedding(
            config["d_model"] // config["num_heads"],
            config["max_seq_length"],
        )
    return MultiHeadAttention(
        config["d_model"],
        config["num_heads"],
        dropout=config["dropout"],
        rope=rope,
    )


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        config: dict[str, Any],
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = config["d_model"]
        self.embedding = nn.Embedding(
            vocab_size, self.d_model, padding_idx=padding_idx
        )
        self.position = (
            SinusoidalPositionalEncoding(
                self.d_model, config["max_seq_length"]
            )
            if config["positional"] == "sinusoidal"
            else None
        )
        norm_class = _norm_class(config)
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    self.d_model,
                    _make_attention(config),
                    norm_class,
                    config["d_ff"],
                    config["dropout"],
                )
                for _ in range(config["num_layers"])
            ]
        )
        self.embedding_dropout = nn.Dropout(config["dropout"])
        self.final_norm = norm_class(self.d_model)

    def forward(
        self,
        source_tokens: torch.Tensor,
        source_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedding(source_tokens) * math.sqrt(self.d_model)
        if self.position is not None:
            x = self.position(x)
        x = self.embedding_dropout(x)
        for block in self.blocks:
            x = block(x, key_padding_mask=source_padding_mask)
        return self.final_norm(x)


class DecoderBlock(nn.Module):
    """Pre-LN causal self-attention, cross-attention, and FFN."""

    def __init__(
        self,
        d_model: int,
        self_attention: nn.Module,
        cross_attention: nn.Module,
        norm_class: type[nn.Module],
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.self_attention_norm = norm_class(d_model)
        self.cross_attention_norm = norm_class(d_model)
        self.ffn_norm = norm_class(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_memory: torch.Tensor,
        target_padding_mask: torch.Tensor | None = None,
        source_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.self_attention_norm(x)
        attended, _ = self.self_attention(
            normalized,
            key_padding_mask=target_padding_mask,
            causal=True,
        )
        x = x + self.dropout(attended)

        attended, _ = self.cross_attention(
            self.cross_attention_norm(x),
            key=encoder_memory,
            value=encoder_memory,
            key_padding_mask=source_padding_mask,
        )
        x = x + self.dropout(attended)
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        config: dict[str, Any],
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = config["d_model"]
        self.embedding = nn.Embedding(
            vocab_size, self.d_model, padding_idx=padding_idx
        )
        self.position = (
            SinusoidalPositionalEncoding(
                self.d_model, config["max_seq_length"]
            )
            if config["positional"] == "sinusoidal"
            else None
        )
        norm_class = _norm_class(config)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    self.d_model,
                    _make_attention(config),
                    # Encoder/decoder positions use different coordinate systems,
                    # so cross-attention does not apply one shared rotary frame.
                    _make_attention(config, use_rope=False),
                    norm_class,
                    config["d_ff"],
                    config["dropout"],
                )
                for _ in range(config["num_layers"])
            ]
        )
        self.embedding_dropout = nn.Dropout(config["dropout"])
        self.final_norm = norm_class(self.d_model)

    def forward(
        self,
        target_tokens: torch.Tensor,
        encoder_memory: torch.Tensor,
        target_padding_mask: torch.Tensor | None = None,
        source_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedding(target_tokens) * math.sqrt(self.d_model)
        if self.position is not None:
            x = self.position(x)
        x = self.embedding_dropout(x)
        for block in self.blocks:
            x = block(
                x,
                encoder_memory,
                target_padding_mask,
                source_padding_mask,
            )
        return self.final_norm(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        config: dict[str, Any],
        source_padding_idx: int | None = None,
        target_padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            source_vocab_size, config, source_padding_idx
        )
        self.decoder = Decoder(
            target_vocab_size, config, target_padding_idx
        )
        # Decoder states predict plaintext ASCII/control-token IDs.
        self.output_projection = nn.Linear(
            config["d_model"], target_vocab_size
        )

    def forward(
        self,
        source_tokens: torch.Tensor,
        decoder_input: torch.Tensor,
        source_padding_mask: torch.Tensor | None = None,
        target_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self.encoder(source_tokens, source_padding_mask)
        hidden = self.decoder(
            decoder_input,
            memory,
            target_padding_mask,
            source_padding_mask,
        )
        return self.output_projection(hidden)


def get_config(config_name: str) -> dict[str, Any]:
    name = config_name.upper()
    if name not in ARCHITECTURE_CONFIGS:
        raise ValueError("Configuration must be C1, C2, C3, or C4.")
    config = {**BASE_CONFIG, **ARCHITECTURE_CONFIGS[name], "name": name}
    if config["d_model"] % config["num_heads"]:
        raise ValueError("d_model must be divisible by num_heads.")
    if name == "C2" and (config["d_model"] // config["num_heads"]) % 2:
        raise ValueError("RoPE requires an even attention head dimension.")
    if name == "C3" and config["num_heads"] % config["num_kv_heads"]:
        raise ValueError("GQA query heads must be divisible by KV heads.")
    return config


def build_model(
    config: dict[str, Any],
    source_vocab_size: int,
    target_vocab_size: int,
    source_padding_idx: int | None = None,
    target_padding_idx: int | None = None,
) -> Seq2SeqTransformer:
    return Seq2SeqTransformer(
        source_vocab_size,
        target_vocab_size,
        config,
        source_padding_idx,
        target_padding_idx,
    )


def compute_batch_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    target_pad_id: int,
) -> tuple[torch.Tensor, int, int]:
    source = batch["src_tokens"].to(device, non_blocking=True)
    source_padding = batch["src_padding_mask"].to(device, non_blocking=True)
    targets = batch["target_tokens"].to(device, non_blocking=True)
    target_padding = batch["target_padding_mask"][:, :-1].to(
        device, non_blocking=True
    )
    decoder_input, labels = targets[:, :-1], targets[:, 1:]

    logits = model(
        source,
        decoder_input,
        source_padding_mask=source_padding,
        target_padding_mask=target_padding,
    )
    valid = labels.ne(target_pad_id)
    token_count = int(valid.sum().item())
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=target_pad_id,
        reduction="sum",
    ) / token_count
    correct = int((logits.argmax(dim=-1).eq(labels) & valid).sum().item())
    return loss, token_count, correct


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_pad_id: int,
    gradient_clip: float,
    run,
    global_step: int,
    log_every: int,
) -> tuple[float, int, int]:
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        loss, token_count, _ = compute_batch_loss(
            model, batch, device, target_pad_id
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        total_loss += float(loss.item()) * token_count
        total_tokens += token_count
        global_step += 1
        if log_every > 0 and global_step % log_every == 0:
            log_training_step(
                run,
                global_step,
                float(loss.item()),
                optimizer.param_groups[0]["lr"],
            )

    return total_loss / total_tokens, global_step, total_tokens


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    target_pad_id: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    for batch in dataloader:
        loss, token_count, correct = compute_batch_loss(
            model, batch, device, target_pad_id
        )
        total_loss += float(loss.item()) * token_count
        total_tokens += token_count
        correct_tokens += correct

    return total_loss / total_tokens, correct_tokens / total_tokens


@torch.no_grad()
def greedy_decode(
    model: Seq2SeqTransformer,
    source_tokens: torch.Tensor,
    source_padding_mask: torch.Tensor,
    bos_token_id: int,
    eos_token_id: int,
    max_length: int,
) -> torch.Tensor:
    model.eval()
    memory = model.encoder(source_tokens, source_padding_mask)
    generated = torch.full(
        (source_tokens.size(0), 1),
        bos_token_id,
        dtype=torch.long,
        device=source_tokens.device,
    )
    finished = torch.zeros(
        source_tokens.size(0), dtype=torch.bool, device=source_tokens.device
    )

    for _ in range(max_length - 1):
        hidden = model.decoder(
            generated,
            memory,
            source_padding_mask=source_padding_mask,
        )
        next_token = model.output_projection(hidden[:, -1]).argmax(dim=-1)
        next_token = torch.where(
            finished, torch.full_like(next_token, eos_token_id), next_token
        )
        generated = torch.cat((generated, next_token.unsqueeze(1)), dim=1)
        finished |= next_token.eq(eos_token_id)
        if bool(finished.all()):
            break
    return generated


@torch.no_grad()
def evaluate_model(
    model: Seq2SeqTransformer,
    test_loader: DataLoader,
    device: torch.device,
    target_codec: PlaintextCodec,
    max_length: int,
) -> dict[str, float]:
    predictions: list[str] = []
    targets: list[str] = []
    for batch in test_loader:
        source = batch["src_tokens"].to(device, non_blocking=True)
        source_padding = batch["src_padding_mask"].to(
            device, non_blocking=True
        )
        generated = greedy_decode(
            model,
            source,
            source_padding,
            target_codec.BOS,
            target_codec.EOS,
            max_length,
        )
        predictions.extend(
            target_codec.decode(row.tolist()) for row in generated.cpu()
        )
        targets.extend(
            target_codec.decode(row.tolist())
            for row in batch["target_tokens"]
        )
    return compute_evaluation_metrics(predictions, targets)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(
    config_name: str, overrides: dict[str, Any] | None = None
) -> dict[str, float]:
    config = get_config(config_name)
    supplied = overrides or {}
    config.update({key: value for key, value in supplied.items() if value is not None})
    if (
        supplied.get("max_seq_length") is not None
        and supplied.get("max_decode_length") is None
    ):
        config["max_decode_length"] = config["max_seq_length"]
    if config["epochs"] <= 0 or config["batch_size"] <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    if config["max_decode_length"] > config["max_seq_length"]:
        raise ValueError("max_decode_length cannot exceed max_seq_length.")

    device = _resolve_device(config["device"])
    config["device"] = str(device)
    _set_seed(config["seed"])

    source_tokenizer = BinaryBPE.load(Path(config["merge_rules_path"]))
    loaders, target_codec, source_pad_id = build_dataloaders(
        source_tokenizer,
        config["batch_size"],
        plain_path=Path(config["plain_path"]),
        cipher_path=Path(config["cipher_path"]),
        train_ratio=config["train_ratio"],
        validation_ratio=config["validation_ratio"],
        test_ratio=config["test_ratio"],
        seed=config["seed"],
        max_seq_length=config["max_seq_length"],
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
    )
    model = build_model(
        config,
        source_vocab_size=len(source_tokenizer.vocabulary) + 1,
        target_vocab_size=target_codec.vocabulary_size,
        source_padding_idx=source_pad_id,
        target_padding_idx=target_codec.PAD,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    run = initialize_wandb(
        config["name"],
        config,
        project=config["wandb_project"],
        mode=config["wandb_mode"],
    )

    checkpoint_dir = Path(config["output_dir"]) / "checkpoints" / config["name"]
    best_path = checkpoint_dir / "best.pt"
    final_path = checkpoint_dir / "final.pt"
    history: dict[str, list[float]] = {
        "train_loss": [],
        "validation_loss": [],
    }
    best_validation_loss = float("inf")
    latest_validation_loss = float("inf")
    global_step = 0
    training_peak_memory = 0.0
    training_started = start_timer()

    for epoch in range(1, config["epochs"] + 1):
        epoch_started = start_timer()
        reset_gpu_peak_memory(device)
        training_epoch_started = start_timer()
        train_loss, global_step, trained_tokens = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            target_codec.PAD,
            config["gradient_clip"],
            run,
            global_step,
            config["log_every"],
        )
        training_epoch_seconds = elapsed_seconds(training_epoch_started)
        validation_loss, validation_accuracy = validate(
            model,
            loaders["validation"],
            device,
            target_codec.PAD,
        )
        latest_validation_loss = validation_loss
        epoch_seconds = elapsed_seconds(epoch_started)
        total_seconds = elapsed_seconds(training_started)
        epoch_peak_memory = gpu_peak_memory_mb(device)
        training_peak_memory = max(training_peak_memory, epoch_peak_memory)
        tokens_per_second = trained_tokens / training_epoch_seconds

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        log_epoch(
            run,
            global_step,
            epoch,
            train_loss,
            validation_loss,
            validation_accuracy,
            optimizer.param_groups[0]["lr"],
            epoch_seconds,
            total_seconds,
            tokens_per_second,
            epoch_peak_memory,
        )
        print(
            f"epoch {epoch:03d} | train {train_loss:.4f} | "
            f"validation {validation_loss:.4f} | "
            f"token accuracy {validation_accuracy:.4f} | "
            f"{tokens_per_second:.0f} tokens/s"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                config,
                {
                    "validation_loss": validation_loss,
                    "validation_token_accuracy": validation_accuracy,
                },
            )

    training_seconds = elapsed_seconds(training_started)
    save_checkpoint(
        final_path,
        model,
        optimizer,
        config["epochs"],
        config,
        {"validation_loss": latest_validation_loss},
    )
    plot_training_history(
        history,
        Path(config["output_dir"]) / "plots" / f"{config['name']}_loss.png",
    )

    results = {
        "best_validation_loss": best_validation_loss,
        "training_time_seconds": training_seconds,
        "gpu_peak_memory_mb": training_peak_memory,
    }
    if config["evaluate_test"]:
        load_checkpoint(best_path, model, device=device)
        test_metrics = evaluate_model(
            model,
            loaders["test"],
            device,
            target_codec,
            config["max_decode_length"],
        )
        test_metrics["training_time_seconds"] = training_seconds
        test_metrics["gpu_peak_memory_mb"] = training_peak_memory
        log_final_evaluation(run, test_metrics)
        results.update(test_metrics)
        print("Final test metrics:")
        for name, value in test_metrics.items():
            print(f"  {name}: {value:.6f}")

    if run is not None:
        run.finish()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", choices=tuple(ARCHITECTURE_CONFIGS))
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--max-decode-length", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--wandb-project")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled")
    )
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--plain-path", type=Path)
    parser.add_argument("--cipher-path", type=Path)
    parser.add_argument("--merge-rules-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "config" and value is not None
    }
    run_training(args.config, overrides)


if __name__ == "__main__":
    main()
