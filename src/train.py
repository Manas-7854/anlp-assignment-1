"""Train the C1-C5 encoder-decoder configurations."""

from __future__ import annotations

import argparse
import json
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
    ByteCodec,
    PlaintextCodec,
    build_blt_dataloaders,
    build_dataloaders,
)
from models.attention import GroupedQueryAttention, MultiHeadAttention
from models.blt import BLTLocalDecoder, BLTLocalEncoder
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

BASE_CONFIG = {
    "d_model": 256,
    "num_layers": 4,
    "num_heads": 8,
    "num_kv_heads": 2,
    "d_ff": 1024,
    "dropout": 0.1,
    "batch_size": 2,
    "learning_rate": 3e-4,
    "min_learning_rate": 1e-5,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "fp16": True,
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
    "overfit_examples": 0,
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
        "tokenization": "bpe",
    },
    "C2": {
        "attention": "mha",
        "positional": "rope",
        "norm": "layernorm",
        "tokenization": "bpe",
    },
    "C3": {
        "attention": "gqa",
        "positional": "sinusoidal",
        "norm": "layernorm",
        "tokenization": "bpe",
    },
    "C4": {
        "attention": "mha",
        "positional": "sinusoidal",
        "norm": "rmsnorm",
        "tokenization": "bpe",
    },
    "C5": {
        "attention": "mha",
        "positional": "sinusoidal",
        "norm": "layernorm",
        "tokenization": "blt",
        "patch_size": 4,
        "local_d_model": 128,
        "local_num_heads": 4,
        "local_d_ff": 512,
        "local_num_layers": 1,
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


def _initialize_embedding(
    embedding: nn.Embedding, d_model: int, padding_idx: int | None
) -> None:
    """Initialize embeddings so scaling by sqrt(d_model) gives unit variance."""

    nn.init.normal_(embedding.weight, mean=0.0, std=d_model**-0.5)
    if padding_idx is not None:
        with torch.no_grad():
            embedding.weight[padding_idx].zero_()


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
        _initialize_embedding(self.embedding, self.d_model, padding_idx)
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
        _initialize_embedding(self.embedding, self.d_model, padding_idx)
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

    def encode_source(
        self,
        source_tokens: torch.Tensor,
        source_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(source_tokens, source_padding_mask), source_padding_mask

    def decode_from_memory(
        self,
        decoder_input: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        target_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.decoder(
            decoder_input,
            memory,
            target_padding_mask,
            memory_padding_mask,
        )
        return self.output_projection(hidden)

    def forward(
        self,
        source_tokens: torch.Tensor,
        decoder_input: torch.Tensor,
        source_padding_mask: torch.Tensor | None = None,
        target_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory, memory_padding_mask = self.encode_source(
            source_tokens, source_padding_mask
        )
        return self.decode_from_memory(
            decoder_input,
            memory,
            memory_padding_mask,
            target_padding_mask,
        )


class BLTSeq2SeqTransformer(nn.Module):
    """C5: local byte models around the same global C1 Transformer."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        local_arguments = {
            "byte_vocab_size": ByteCodec.vocabulary_size,
            "padding_idx": ByteCodec.PAD,
            "local_d_model": config["local_d_model"],
            "global_d_model": config["d_model"],
            "num_heads": config["local_num_heads"],
            "d_ff": config["local_d_ff"],
            "num_layers": config["local_num_layers"],
            "patch_size": config["patch_size"],
            "max_seq_length": config["max_seq_length"],
            "dropout": config["dropout"],
        }
        self.source_local_encoder = BLTLocalEncoder(**local_arguments)
        self.target_patch_encoder = BLTLocalEncoder(**local_arguments)
        self.local_decoder = BLTLocalDecoder(**local_arguments)
        self.patch_size = config["patch_size"]
        self.bos_patch = nn.Parameter(torch.zeros(1, 1, config["d_model"]))

        norm_class = _norm_class(config)
        self.global_position = SinusoidalPositionalEncoding(
            config["d_model"], config["max_seq_length"]
        )
        self.global_dropout = nn.Dropout(config["dropout"])
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    config["d_model"],
                    _make_attention(config),
                    norm_class,
                    config["d_ff"],
                    config["dropout"],
                )
                for _ in range(config["num_layers"])
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    config["d_model"],
                    _make_attention(config),
                    _make_attention(config, use_rope=False),
                    norm_class,
                    config["d_ff"],
                    config["dropout"],
                )
                for _ in range(config["num_layers"])
            ]
        )
        self.encoder_norm = norm_class(config["d_model"])
        self.decoder_norm = norm_class(config["d_model"])

    def encode_source(
        self,
        source_tokens: torch.Tensor,
        source_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, patch_mask = self.source_local_encoder(
            source_tokens, source_padding_mask
        )
        x = self.global_dropout(self.global_position(x))
        for block in self.encoder_blocks:
            x = block(x, key_padding_mask=patch_mask)
        return self.encoder_norm(x), patch_mask

    def decode_from_memory(
        self,
        decoder_input: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        target_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length = decoder_input.shape
        output_patch_mask = self.target_patch_encoder.patch_padding_mask(
            target_padding_mask
        )
        if length > 1:
            previous_patches, _ = self.target_patch_encoder(
                decoder_input[:, 1:], target_padding_mask[:, 1:]
            )
        else:
            previous_patches = memory.new_empty(batch, 0, memory.size(-1))

        # Patch j sees only completed target patch j-1, never its own labels.
        bos = self.bos_patch.expand(batch, -1, -1)
        x = torch.cat((bos, previous_patches), dim=1)
        x = x[:, : output_patch_mask.size(1)]
        x = self.global_dropout(self.global_position(x))
        for block in self.decoder_blocks:
            x = block(
                x,
                memory,
                target_padding_mask=output_patch_mask,
                source_padding_mask=memory_padding_mask,
            )
        patch_states = self.decoder_norm(x)
        return self.local_decoder(
            decoder_input, patch_states, target_padding_mask
        )

    def forward(
        self,
        source_tokens: torch.Tensor,
        decoder_input: torch.Tensor,
        source_padding_mask: torch.Tensor,
        target_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory, memory_padding_mask = self.encode_source(
            source_tokens, source_padding_mask
        )
        return self.decode_from_memory(
            decoder_input,
            memory,
            memory_padding_mask,
            target_padding_mask,
        )


def get_config(config_name: str) -> dict[str, Any]:
    name = config_name.upper()
    if name not in ARCHITECTURE_CONFIGS:
        raise ValueError("Configuration must be C1, C2, C3, C4, or C5.")
    config = {**BASE_CONFIG, **ARCHITECTURE_CONFIGS[name], "name": name}
    if config["d_model"] % config["num_heads"]:
        raise ValueError("d_model must be divisible by num_heads.")
    if name == "C2" and (config["d_model"] // config["num_heads"]) % 2:
        raise ValueError("RoPE requires an even attention head dimension.")
    if name == "C3" and config["num_heads"] % config["num_kv_heads"]:
        raise ValueError("GQA query heads must be divisible by KV heads.")
    if name == "C5":
        if config["local_d_model"] % config["local_num_heads"]:
            raise ValueError("C5 local_d_model must be divisible by local heads.")
        if config["patch_size"] <= 0:
            raise ValueError("C5 patch_size must be positive.")
    return config


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
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    target_pad_id: int,
    gradient_clip: float,
    run,
    global_step: int,
    log_every: int,
    use_fp16: bool,
) -> tuple[float, int, int]:
    model.train()
    total_loss = 0.0
    total_tokens = 0

    total_batches = len(dataloader)
    progress_every = max(1, total_batches // 4)
    for batch_number, batch in enumerate(dataloader, start=1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_fp16
        ):
            loss, token_count, _ = compute_batch_loss(
                model, batch, device, target_pad_id
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scale_before_update = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scale_before_update:
            scheduler.step()

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
        if batch_number % progress_every == 0 or batch_number == total_batches:
            print(
                f"  training: batch {batch_number}/{total_batches}", flush=True
            )

    return total_loss / total_tokens, global_step, total_tokens


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    target_pad_id: int,
    use_fp16: bool,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    for batch in dataloader:
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_fp16
        ):
            loss, token_count, correct = compute_batch_loss(
                model, batch, device, target_pad_id
            )
        total_loss += float(loss.item()) * token_count
        total_tokens += token_count
        correct_tokens += correct

    return total_loss / total_tokens, correct_tokens / total_tokens


@torch.no_grad()
def greedy_decode(
    model: Seq2SeqTransformer | BLTSeq2SeqTransformer,
    source_tokens: torch.Tensor,
    source_padding_mask: torch.Tensor,
    bos_token_id: int,
    eos_token_id: int,
    max_length: int,
) -> torch.Tensor:
    model.eval()
    use_fp16 = source_tokens.device.type == "cuda"
    with torch.autocast(
        device_type=source_tokens.device.type,
        dtype=torch.float16,
        enabled=use_fp16,
    ):
        memory, memory_padding_mask = model.encode_source(
            source_tokens, source_padding_mask
        )
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
        with torch.autocast(
            device_type=source_tokens.device.type,
            dtype=torch.float16,
            enabled=use_fp16,
        ):
            logits = model.decode_from_memory(
                generated,
                memory,
                memory_padding_mask,
                torch.zeros_like(generated, dtype=torch.bool),
            )
            next_token = logits[:, -1].argmax(dim=-1)
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
    model: Seq2SeqTransformer | BLTSeq2SeqTransformer,
    test_loader: DataLoader,
    device: torch.device,
    target_codec: PlaintextCodec | ByteCodec,
    max_length: int,
    include_tokenized_metrics: bool,
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
    return compute_evaluation_metrics(
        predictions,
        targets,
        include_tokenized_metrics=include_tokenized_metrics,
    )


def dataset_statistics(loaders: dict[str, DataLoader]) -> dict[str, Any]:
    """Return auditable split sizes and encoded sequence-length ranges."""

    statistics: dict[str, Any] = {}
    for split_name, loader in loaders.items():
        examples = loader.dataset.examples
        source_lengths = [int(source.numel()) for source, _ in examples]
        target_lengths = [int(target.numel()) for _, target in examples]
        statistics[split_name] = {
            "examples": len(examples),
            "batches_per_epoch": len(loader),
            "source_min_length": min(source_lengths) if source_lengths else 0,
            "source_max_length": max(source_lengths) if source_lengths else 0,
            "target_min_length": min(target_lengths) if target_lengths else 0,
            "target_max_length": max(target_lengths) if target_lengths else 0,
        }
    return statistics


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


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
    min_learning_rate: float,
) -> tuple[torch.optim.lr_scheduler.LambdaLR, int]:
    """Create a linear-warmup then cosine-decay per-step LR schedule."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    base_learning_rate = optimizer.param_groups[0]["lr"]
    minimum_ratio = min_learning_rate / base_learning_rate
    warmup_steps = min(total_steps, round(total_steps * warmup_ratio))
    decay_steps = max(1, total_steps - warmup_steps)

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier), warmup_steps


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
    if not 0.0 <= config["min_learning_rate"] <= config["learning_rate"]:
        raise ValueError(
            "min_learning_rate must be between zero and learning_rate."
        )
    if not 0.0 <= config["warmup_ratio"] < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1).")
    if config["overfit_examples"] < 0:
        raise ValueError("overfit_examples cannot be negative.")
    if config["max_decode_length"] > config["max_seq_length"]:
        raise ValueError("max_decode_length cannot exceed max_seq_length.")

    device = _resolve_device(config["device"])
    config["device"] = str(device)
    use_fp16 = bool(config["fp16"] and device.type == "cuda")
    _set_seed(config["seed"])
    print(
        f"Starting {config['name']} on {device} "
        f"({config['epochs']} epochs, batch size {config['batch_size']})",
        flush=True,
    )
    print(
        f"Precision: {'FP16 mixed precision' if use_fp16 else 'FP32'}",
        flush=True,
    )

    data_arguments = {
        "batch_size": config["batch_size"],
        "plain_path": Path(config["plain_path"]),
        "cipher_path": Path(config["cipher_path"]),
        "train_ratio": config["train_ratio"],
        "validation_ratio": config["validation_ratio"],
        "test_ratio": config["test_ratio"],
        "seed": config["seed"],
        "max_seq_length": config["max_seq_length"],
        "num_workers": config["num_workers"],
        "pin_memory": device.type == "cuda",
        "overfit_examples": config["overfit_examples"],
    }
    print("Loading and encoding dataset splits...", flush=True)
    if config["tokenization"] == "blt":
        loaders, target_codec, _ = build_blt_dataloaders(
            **data_arguments
        )
        model = BLTSeq2SeqTransformer(config)
        config["source_vocab_size"] = ByteCodec.vocabulary_size
        config["target_vocab_size"] = ByteCodec.vocabulary_size
    else:
        print("Loading frozen BPE tokenizer...", flush=True)
        source_tokenizer = BinaryBPE.load(Path(config["merge_rules_path"]))
        loaders, target_codec, source_pad_id = build_dataloaders(
            source_tokenizer, **data_arguments
        )
        model = Seq2SeqTransformer(
            len(source_tokenizer.vocabulary) + 1,
            target_codec.vocabulary_size,
            config,
            source_pad_id,
            target_codec.PAD,
        )
        config["source_vocab_size"] = len(source_tokenizer.vocabulary) + 1
        config["target_vocab_size"] = target_codec.vocabulary_size
    if config["overfit_examples"]:
        example_count = len(loaders["train"].dataset)
        config["name"] = f"{config['name']}_overfit_{example_count}"
        config["evaluate_test"] = False
        print(
            f"Overfit diagnostic: training and validating on the same "
            f"{example_count} examples.",
            flush=True,
        )
    print(
        "Dataset ready: "
        + ", ".join(f"{name}={len(loader.dataset)}" for name, loader in loaders.items()),
        flush=True,
    )
    config["dataset_statistics"] = dataset_statistics(loaders)
    print("Building model and optimizer...", flush=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    total_training_steps = config["epochs"] * len(loaders["train"])
    scheduler, warmup_steps = build_lr_scheduler(
        optimizer,
        total_training_steps,
        config["warmup_ratio"],
        config["min_learning_rate"],
    )
    print(
        "Learning-rate schedule: linear warmup + cosine decay; "
        f"{warmup_steps} warmup steps, peak {config['learning_rate']:.2e}, "
        f"minimum "
        f"{config['min_learning_rate']:.2e} over "
        f"{total_training_steps} steps",
        flush=True,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config["parameter_count"] = parameter_count
    print(f"Model ready ({parameter_count:,} parameters).", flush=True)
    print(f"Initializing WandB ({config['wandb_mode']} mode)...", flush=True)
    run = initialize_wandb(
        config["name"],
        config,
        project=config["wandb_project"],
        mode=config["wandb_mode"],
    )
    print("WandB initialized. Starting training...", flush=True)

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
        print(f"Epoch {epoch}/{config['epochs']}: training...", flush=True)
        epoch_started = start_timer()
        reset_gpu_peak_memory(device)
        training_epoch_started = start_timer()
        train_loss, global_step, trained_tokens = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            scheduler,
            scaler,
            device,
            target_codec.PAD,
            config["gradient_clip"],
            run,
            global_step,
            config["log_every"],
            use_fp16,
        )
        training_epoch_seconds = elapsed_seconds(training_epoch_started)
        print(f"Epoch {epoch}/{config['epochs']}: validating...", flush=True)
        validation_loss, validation_accuracy = validate(
            model,
            loaders["validation"],
            device,
            target_codec.PAD,
            use_fp16,
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
        best_checkpoint = load_checkpoint(best_path, model, device=device)
        test_loss, test_token_accuracy = validate(
            model,
            loaders["test"],
            device,
            target_codec.PAD,
            use_fp16,
        )
        test_metrics = evaluate_model(
            model,
            loaders["test"],
            device,
            target_codec,
            config["max_decode_length"],
            include_tokenized_metrics=config["tokenization"] == "bpe",
        )
        test_metrics["loss"] = test_loss
        test_metrics["token_accuracy"] = test_token_accuracy
        test_metrics["training_time_seconds"] = training_seconds
        test_metrics["gpu_peak_memory_mb"] = training_peak_memory
        log_final_evaluation(run, test_metrics)
        results.update(test_metrics)
        metrics_path = (
            Path(config["output_dir"])
            / "metrics"
            / f"{config['name']}_test_metrics.json"
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_payload = {
            "configuration": config["name"],
            "evaluation": "best validation checkpoint; greedy decoding",
            "best_checkpoint_epoch": best_checkpoint["epoch"],
            "test_metrics": test_metrics,
            "hyperparameters": config,
        }
        metrics_path.write_text(
            json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Saved final test metrics to {metrics_path}")
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
    parser.add_argument("--min-learning-rate", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument(
        "--fp16", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--max-decode-length", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--wandb-project")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled")
    )
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--overfit-examples", type=int)
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
