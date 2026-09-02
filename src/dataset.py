"""Dataset loading, batching, splitting, and binary BPE tokenization."""

from __future__ import annotations

import argparse
import json
import random
from functools import partial
from pathlib import Path
from typing import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
PLAIN_PATH = ROOT / "src" / "datset" / "brown_plain.txt"
CIPHER_PATH = ROOT / "src" / "datset" / "brown_cipher.txt"
MERGE_RULES_PATH = ROOT / "outputs" / "merge_rules.json"
BOUNDARY = -1


def read_dataset(
    plain_path: Path = PLAIN_PATH,
    cipher_path: Path = CIPHER_PATH,
) -> list[tuple[str, str]]:
    """Read and validate line-aligned plaintext/ciphertext pairs."""

    plain_lines = plain_path.read_text(encoding="ascii").splitlines()
    cipher_lines = cipher_path.read_text(encoding="ascii").splitlines()

    if len(plain_lines) != len(cipher_lines):
        raise ValueError("Plaintext and ciphertext files have different line counts.")

    for line_number, (plain, cipher) in enumerate(
        zip(plain_lines, cipher_lines), start=1
    ):
        if not cipher or not set(cipher) <= {"0", "1"}:
            raise ValueError(f"Ciphertext line {line_number} is not binary.")
        if len(cipher) != 8 * len(plain):
            raise ValueError(f"Line {line_number} is not aligned by length.")

    return list(zip(plain_lines, cipher_lines))


def split_dataset(
    pairs: Sequence[tuple[str, str]],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[tuple[str, str]]]:
    """Shuffle reproducibly and split aligned pairs without writing extra files."""

    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio <= 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Split ratios must be positive and sum to 1.0.")

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)

    train_end = int(len(shuffled) * train_ratio)
    validation_end = train_end + int(len(shuffled) * validation_ratio)
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }


def _flatten(sequences: Sequence[str], device: str) -> torch.Tensor:
    """Flatten binary sequences with -1 boundaries so merges never cross lines."""

    if not hasattr(torch, "empty"):
        raise RuntimeError("PyTorch is not installed correctly in this environment.")

    flat = torch.empty(
        sum(len(sequence) + 1 for sequence in sequences), dtype=torch.long
    )
    position = 0

    for sequence in sequences:
        if not sequence or not set(sequence) <= {"0", "1"}:
            raise ValueError("BPE accepts only non-empty binary sequences.")

        bits = torch.frombuffer(
            bytearray(sequence, encoding="ascii"), dtype=torch.uint8
        ).long()
        end = position + len(sequence)
        flat[position:end] = bits - ord("0")
        flat[end] = BOUNDARY
        position = end + 1

    return flat.to(device)


def _most_frequent_pair(
    tokens: torch.Tensor, vocabulary_size: int
) -> tuple[int, int, int] | None:
    """Return the most frequent adjacent pair, breaking ties by token IDs."""

    left = tokens[:-1]
    right = tokens[1:]
    valid = (left >= 0) & (right >= 0)
    if not torch.any(valid):
        return None

    pair_codes = left[valid] * vocabulary_size + right[valid]
    counts = torch.bincount(pair_codes)
    pair_code = int(torch.argmax(counts).item())
    return (
        pair_code // vocabulary_size,
        pair_code % vocabulary_size,
        int(counts[pair_code].item()),
    )


def _merge_pair(
    tokens: torch.Tensor,
    left_id: int,
    right_id: int,
    new_id: int,
) -> tuple[torch.Tensor, int]:
    """Replace non-overlapping occurrences of one pair from left to right."""

    matches = (tokens[:-1] == left_id) & (tokens[1:] == right_id)
    positions = torch.nonzero(matches, as_tuple=False).flatten()
    if positions.numel() == 0:
        return tokens, 0

    # Pairs such as (0, 0) overlap inside runs like 0000.
    if left_id == right_id and positions.numel() > 1:
        starts = torch.ones(
            positions.numel(), dtype=torch.bool, device=positions.device
        )
        starts[1:] = positions[1:] != positions[:-1] + 1
        run_ids = torch.cumsum(starts.long(), dim=0) - 1
        offsets = positions - positions[starts][run_ids]
        positions = positions[offsets.remainder(2) == 0]

    merged = tokens.clone()
    merged[positions] = new_id
    remove = torch.zeros(tokens.numel(), dtype=torch.bool, device=tokens.device)
    remove[positions + 1] = True
    return merged[~remove], int(positions.numel())


class BinaryBPE:
    """Binary BPE represented by ordered adjacent-token merge rules."""

    def __init__(self, merges: Sequence[tuple[int, int]] = ()) -> None:
        self.merges = [(int(left), int(right)) for left, right in merges]
        self.vocabulary = ["0", "1"]

        for new_id, (left_id, right_id) in enumerate(self.merges, start=2):
            if left_id >= new_id or right_id >= new_id:
                raise ValueError("A merge rule refers to a token not learned yet.")
            self.vocabulary.append(
                self.vocabulary[left_id] + self.vocabulary[right_id]
            )

    @classmethod
    def train(
        cls,
        sequences: Sequence[str],
        number_of_merges: int,
        device: str = "cpu",
        minimum_frequency: int = 2,
    ) -> "BinaryBPE":
        """Repeatedly merge the most frequent adjacent training-corpus pair."""

        if not sequences:
            raise ValueError("The BPE training split is empty.")
        if number_of_merges < 0:
            raise ValueError("number_of_merges must be non-negative.")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        tokens = _flatten(sequences, device)
        raw_bits = int((tokens >= 0).sum().item())
        merges: list[tuple[int, int]] = []

        for step in range(number_of_merges):
            pair = _most_frequent_pair(tokens, len(merges) + 2)
            if pair is None or pair[2] < minimum_frequency:
                break

            left_id, right_id, frequency = pair
            new_id = len(merges) + 2
            tokens, replacements = _merge_pair(
                tokens, left_id, right_id, new_id
            )
            merges.append((left_id, right_id))

            if step == 0 or (step + 1) % 10 == 0 or step + 1 == number_of_merges:
                token_count = int((tokens >= 0).sum().item())
                print(
                    f"merge {step + 1}/{number_of_merges}: "
                    f"pair=({left_id}, {right_id}), frequency={frequency}, "
                    f"replacements={replacements}, "
                    f"compression={raw_bits / token_count:.3f}x"
                )

        return cls(merges)

    def encode(self, sequence: str) -> list[int]:
        """Encode one binary string using the frozen merge order."""

        if not sequence or not set(sequence) <= {"0", "1"}:
            raise ValueError("encode() expects a non-empty binary string.")

        token_ids = [int(bit) for bit in sequence]
        for new_id, (left_id, right_id) in enumerate(self.merges, start=2):
            output: list[int] = []
            position = 0
            while position < len(token_ids):
                if (
                    position + 1 < len(token_ids)
                    and token_ids[position] == left_id
                    and token_ids[position + 1] == right_id
                ):
                    output.append(new_id)
                    position += 2
                else:
                    output.append(token_ids[position])
                    position += 1
            token_ids = output
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        """Reconstruct the exact bit string represented by token IDs."""

        return "".join(self.vocabulary[token_id] for token_id in token_ids)

    def save(self, path: Path = MERGE_RULES_PATH) -> None:
        """Save ordered merge rules to one human-readable JSON file."""

        path.parent.mkdir(parents=True, exist_ok=True)
        rules = [
            {
                "left_id": left_id,
                "right_id": right_id,
                "new_id": new_id,
                "token": self.vocabulary[new_id],
            }
            for new_id, (left_id, right_id) in enumerate(self.merges, start=2)
        ]
        path.write_text(json.dumps(rules, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path = MERGE_RULES_PATH) -> "BinaryBPE":
        """Load ordered merge rules saved by :meth:`save`."""

        rules = json.loads(path.read_text(encoding="utf-8"))
        return cls([(rule["left_id"], rule["right_id"]) for rule in rules])


class PlaintextCodec:
    """Map ASCII plaintext to decoder IDs with PAD/BOS/EOS control tokens."""

    ASCII_VOCAB_SIZE = 128
    PAD = 128
    BOS = 129
    EOS = 130
    vocabulary_size = 131

    def encode(self, text: str) -> list[int]:
        if any(ord(character) >= self.ASCII_VOCAB_SIZE for character in text):
            raise ValueError("Plaintext contains a non-ASCII character.")
        return [self.BOS, *(ord(character) for character in text), self.EOS]

    def decode(self, token_ids: Sequence[int]) -> str:
        characters: list[str] = []
        for token_id in map(int, token_ids):
            if token_id == self.EOS:
                break
            if token_id in (self.PAD, self.BOS):
                continue
            if 0 <= token_id < self.ASCII_VOCAB_SIZE:
                characters.append(chr(token_id))
        return "".join(characters)


class PairedSequenceDataset(Dataset):
    """Pre-encode aligned ciphertext/plaintext examples for model training."""

    def __init__(
        self,
        pairs: Sequence[tuple[str, str]],
        source_tokenizer: BinaryBPE,
        target_codec: PlaintextCodec,
        max_seq_length: int,
        split_name: str = "dataset",
    ) -> None:
        self.examples: list[tuple[torch.Tensor, torch.Tensor]] = []
        total = len(pairs)
        progress_every = max(1, total // 4)
        print(f"Encoding {split_name} split ({total} examples)...", flush=True)
        for index, (plaintext, ciphertext) in enumerate(pairs):
            source = source_tokenizer.encode(ciphertext)
            target = target_codec.encode(plaintext)
            if max(len(source), len(target)) > max_seq_length:
                raise ValueError(
                    f"Example {index} has source/target lengths "
                    f"{len(source)}/{len(target)}, exceeding max_seq_length="
                    f"{max_seq_length}. Data is not truncated."
                )
            self.examples.append(
                (
                    torch.tensor(source, dtype=torch.long),
                    torch.tensor(target, dtype=torch.long),
                )
            )
            completed = index + 1
            if completed % progress_every == 0 or completed == total:
                print(
                    f"  {split_name}: encoded {completed}/{total} examples",
                    flush=True,
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[index]


def collate_batch(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]],
    source_pad_id: int,
    target_pad_id: int,
) -> dict[str, torch.Tensor]:
    sources, targets = zip(*batch)
    source_tokens = pad_sequence(
        sources, batch_first=True, padding_value=source_pad_id
    )
    target_tokens = pad_sequence(
        targets, batch_first=True, padding_value=target_pad_id
    )
    return {
        "src_tokens": source_tokens,
        "target_tokens": target_tokens,
        "src_padding_mask": source_tokens.eq(source_pad_id),
        "target_padding_mask": target_tokens.eq(target_pad_id),
    }


def build_datasets(
    source_tokenizer: BinaryBPE,
    plain_path: Path = PLAIN_PATH,
    cipher_path: Path = CIPHER_PATH,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    max_seq_length: int = 4096,
) -> tuple[dict[str, PairedSequenceDataset], PlaintextCodec]:
    pairs = read_dataset(plain_path, cipher_path)
    splits = split_dataset(
        pairs, train_ratio, validation_ratio, test_ratio, seed
    )
    target_codec = PlaintextCodec()
    datasets = {
        name: PairedSequenceDataset(
            split,
            source_tokenizer,
            target_codec,
            max_seq_length,
            split_name=name,
        )
        for name, split in splits.items()
    }
    return datasets, target_codec


def build_dataloaders(
    source_tokenizer: BinaryBPE,
    batch_size: int,
    plain_path: Path = PLAIN_PATH,
    cipher_path: Path = CIPHER_PATH,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    max_seq_length: int = 4096,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[dict[str, DataLoader], PlaintextCodec, int]:
    datasets, target_codec = build_datasets(
        source_tokenizer,
        plain_path,
        cipher_path,
        train_ratio,
        validation_ratio,
        test_ratio,
        seed,
        max_seq_length,
    )
    source_pad_id = len(source_tokenizer.vocabulary)
    collate = partial(
        collate_batch,
        source_pad_id=source_pad_id,
        target_pad_id=target_codec.PAD,
    )
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=generator if name == "train" else None,
        )
        for name, dataset in datasets.items()
    }
    return loaders, target_codec, source_pad_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, ...")
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--number-of-merges", type=int)
    size.add_argument("--vocabulary-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = read_dataset()
    splits = split_dataset(
        pairs,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(
        f"Loaded {len(pairs)} aligned pairs; "
        f"train={len(splits['train'])}, "
        f"validation={len(splits['validation'])}, test={len(splits['test'])}."
    )

    if args.vocabulary_size is not None:
        if args.vocabulary_size < 2:
            raise ValueError("vocabulary-size must be at least 2.")
        number_of_merges = args.vocabulary_size - 2
    else:
        number_of_merges = (
            args.number_of_merges if args.number_of_merges is not None else 254
        )

    train_ciphertext = [cipher for _, cipher in splits["train"]]
    tokenizer = BinaryBPE.train(
        train_ciphertext,
        number_of_merges=number_of_merges,
        device=args.device,
    )
    tokenizer.save()
    print(f"Saved {len(tokenizer.merges)} rules to {MERGE_RULES_PATH}.")


if __name__ == "__main__":
    main()
