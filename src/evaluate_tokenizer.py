"""Report simple statistics for the frozen binary BPE tokenizer."""

import json
from collections import Counter
from statistics import fmean

from dataset import BinaryBPE, read_dataset, split_dataset


def split_statistics(
    tokenizer: BinaryBPE, sequences: list[str], top_k: int = 10
) -> dict[str, object]:
    token_counts: Counter[int] = Counter()
    sequence_lengths = []

    for sequence in sequences:
        token_ids = tokenizer.encode(sequence)
        token_counts.update(token_ids)
        sequence_lengths.append(len(token_ids))

    common_learned_tokens = [
        (token_id, count)
        for token_id, count in token_counts.most_common()
        if token_id >= 2
    ][:top_k]

    return {
        "average_tokens_per_sequence": fmean(sequence_lengths),
        "most_frequent_learned_tokens": [
            {
                "id": token_id,
                "token": tokenizer.vocabulary[token_id],
                "bit_length": len(tokenizer.vocabulary[token_id]),
                "count": count,
            }
            for token_id, count in common_learned_tokens
        ],
    }


def main() -> None:
    tokenizer = BinaryBPE.load()
    splits = split_dataset(read_dataset(), seed=42)
    token_lengths = [len(token) for token in tokenizer.vocabulary]

    results = {
        "vocabulary": {
            "size": len(tokenizer.vocabulary),
            "average_token_length_bits": fmean(token_lengths),
            "smallest_token_length_bits": min(token_lengths),
            "largest_token_length_bits": max(token_lengths),
            "token_length_distribution": dict(sorted(Counter(token_lengths).items())),
        },
        "train": split_statistics(
            tokenizer, [cipher for _, cipher in splits["train"]]
        ),
        "test": split_statistics(
            tokenizer, [cipher for _, cipher in splits["test"]]
        ),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
