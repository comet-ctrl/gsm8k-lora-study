"""Stable, disjoint splits: first 200 train rows are development-only."""

import random

DEV_SIZE = 200
SHOT_SIZE = 3
DATASET = "openai/gsm8k"


def select_indices(length: int, split: str, limit: int, seed: int) -> list[int]:
    pools = {
        "train": range(DEV_SIZE + SHOT_SIZE, length),
        "dev": range(min(DEV_SIZE, length)),
        "shots": range(DEV_SIZE, min(DEV_SIZE + SHOT_SIZE, length)),
        "test": range(length),
    }
    indices = list(pools[split])
    if limit < 1 or limit > len(indices):
        raise ValueError(f"Requested {limit} rows from {split}; available: {len(indices)}")
    if split != "shots":
        random.Random(seed).shuffle(indices)
    return indices[:limit]


def load_rows(split: str, limit: int, seed: int, revision: str) -> tuple[list[dict], dict]:
    from datasets import load_dataset
    dataset = load_dataset(DATASET, "main", split="test" if split == "test" else "train", revision=revision)
    indices = select_indices(len(dataset), split, limit, seed)
    rows = [{"id": i, **dataset[i]} for i in indices]
    return rows, {"dataset": DATASET, "revision": revision, "fingerprint": dataset._fingerprint,
                  "split": split, "row_ids": indices, "seed": seed}
