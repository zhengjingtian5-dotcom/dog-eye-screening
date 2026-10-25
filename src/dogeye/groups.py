"""Group near-duplicate images and make a leak-free train/val/test split.

Public datasets often contain the same photo several times (flipped, cropped,
re-encoded, or as Roboflow augmentations). If copies land on both sides of a
split, test accuracy is inflated. We link two images when either their
perceptual hashes are close or their DINOv2 embeddings are nearly identical,
take connected components as groups, and split by group.

Usage:
    python -m dogeye.groups --manifest data/manifest.csv --features data/features.npz \
        --out data/manifest_split.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PHASH_MAX_DISTANCE = 6
COSINE_MIN_SIMILARITY = 0.97
CHUNK = 1024


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def labels(self) -> np.ndarray:
        return np.array([self.find(i) for i in range(len(self.parent))])


def phash_ints(paths: list[str]) -> np.ndarray:
    import imagehash
    from PIL import Image

    return np.array([int(str(imagehash.phash(Image.open(p))), 16) for p in paths], dtype=np.uint64)


def close_hash_pairs(hashes: np.ndarray, max_distance: int = PHASH_MAX_DISTANCE):
    """Yield index pairs (i < j) whose 64-bit hashes differ in at most max_distance bits."""
    for start in range(0, len(hashes), CHUNK):
        block = hashes[start : start + CHUNK, None] ^ hashes[None, :]
        distance = np.bitwise_count(block)
        for i, j in zip(*np.nonzero(distance <= max_distance)):
            if start + i < j:
                yield start + i, j


def similar_embedding_pairs(features: np.ndarray, min_similarity: float = COSINE_MIN_SIMILARITY):
    """Yield index pairs (i < j) whose embeddings have cosine similarity >= min_similarity."""
    unit = features / np.linalg.norm(features, axis=1, keepdims=True).clip(min=1e-12)
    for start in range(0, len(unit), CHUNK):
        similarity = unit[start : start + CHUNK] @ unit.T
        for i, j in zip(*np.nonzero(similarity >= min_similarity)):
            if start + i < j:
                yield start + i, j


def duplicate_groups(n: int, pair_sources) -> np.ndarray:
    uf = UnionFind(n)
    for pairs in pair_sources:
        for i, j in pairs:
            uf.union(int(i), int(j))
    return uf.labels()


def group_split(labels: pd.Series, groups: np.ndarray, test_size: float = 0.15, val_size: float = 0.15, seed: int = 0) -> np.ndarray:
    """Assign each row to train/val/test so no group spans two splits, stratified by label."""
    split = np.full(len(labels), "train", dtype=object)
    y = labels.to_numpy()

    test_folds = StratifiedGroupKFold(n_splits=round(1 / test_size), shuffle=True, random_state=seed)
    rest, test = next(test_folds.split(np.zeros(len(y)), y, groups))
    split[test] = "test"

    val_folds = StratifiedGroupKFold(n_splits=round((1 - test_size) / val_size), shuffle=True, random_state=seed)
    _, val = next(val_folds.split(np.zeros(len(rest)), y[rest], groups[rest]))
    split[rest[val]] = "val"
    return split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--features", default="data/features.npz")
    parser.add_argument("--out", default="data/manifest_split.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    features = np.load(args.features, allow_pickle=True)["features"]
    if len(features) != len(manifest):
        raise SystemExit("features and manifest have different lengths; rerun dogeye.features")

    hashes = phash_ints(manifest["path"].tolist())
    groups = duplicate_groups(len(manifest), [close_hash_pairs(hashes), similar_embedding_pairs(features)])
    manifest["group"] = groups
    manifest["feature_row"] = np.arange(len(manifest))
    manifest = manifest[manifest["label"] != "exclude"].reset_index(drop=True)

    mixed = manifest.groupby("group")["label"].nunique()
    if (mixed > 1).any():
        print(f"warning: {int((mixed > 1).sum())} duplicate groups carry conflicting labels; dropping them")
        manifest = manifest[~manifest["group"].isin(mixed[mixed > 1].index)].reset_index(drop=True)

    manifest["split"] = group_split(manifest["label"], manifest["group"].to_numpy(), seed=args.seed)
    manifest.to_csv(args.out, index=False)

    n_groups = manifest["group"].nunique()
    print(f"{len(manifest)} images in {n_groups} groups ({len(manifest) - n_groups} near-duplicates folded together)")
    print(manifest.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
