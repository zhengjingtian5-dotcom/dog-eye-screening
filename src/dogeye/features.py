"""Extract frozen DINOv2 image embeddings for every row in the manifest.

The embeddings feed both the near-duplicate grouping and the baseline
classifier, so they are computed once and cached to an .npz file.

Usage:
    python -m dogeye.features --manifest data/manifest.csv --out data/features.npz
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from PIL import Image

DEFAULT_MODEL = "facebook/dinov2-small"


def extract(paths: list[str], model_name: str = DEFAULT_MODEL, batch_size: int = 64) -> np.ndarray:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    chunks = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            images = [Image.open(p).convert("RGB") for p in paths[start : start + batch_size]]
            inputs = processor(images=images, return_tensors="pt").to(device)
            chunks.append(model(**inputs).pooler_output.float().cpu().numpy())
            print(f"{min(start + batch_size, len(paths))}/{len(paths)}", end="\r")
    print()
    return np.concatenate(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--out", default="data/features.npz")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    features = extract(manifest["path"].tolist(), args.model, args.batch_size)
    np.savez_compressed(args.out, features=features, paths=manifest["path"].to_numpy(), model=args.model)
    print(f"wrote {features.shape} to {args.out}")


if __name__ == "__main__":
    main()
