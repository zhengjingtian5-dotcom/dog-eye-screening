"""Turn one photo into a screening result the demo can show.

Three outcomes, never a diagnosis:
    see_vet   probability at or above the validation threshold
    uncertain probability just below the threshold, or a photo too dark,
              too bright or too blurry to judge; the advice is to retake the
              photo or see a vet
    normal    clearly below the threshold
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
from PIL import Image

UNCERTAIN_MARGIN = 0.15
MIN_BRIGHTNESS, MAX_BRIGHTNESS = 40, 225
MIN_SHARPNESS = 30.0


@dataclass
class Result:
    outcome: str
    probability: float | None
    threshold: float
    quality_issue: str | None = None


def quality_issue(image: Image.Image) -> str | None:
    """Return "dark", "bright" or "blurry" when the photo is too poor to judge."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    brightness = gray.mean()
    if brightness < MIN_BRIGHTNESS:
        return "dark"
    if brightness > MAX_BRIGHTNESS:
        return "bright"
    # Variance of a 4-neighbour Laplacian: low means few sharp edges.
    laplacian = (
        -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    if laplacian.var() < MIN_SHARPNESS:
        return "blurry"
    return None


def decide(probability: float, threshold: float, margin: float = UNCERTAIN_MARGIN) -> str:
    if probability >= threshold:
        return "see_vet"
    if probability >= threshold - margin:
        return "uncertain"
    return "normal"


def dinov2_embedder(model_name: str = "facebook/dinov2-small") -> Callable[[Image.Image], np.ndarray]:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()

    def embed(image: Image.Image) -> np.ndarray:
        with torch.inference_mode():
            inputs = processor(images=[image.convert("RGB")], return_tensors="pt")
            return model(**inputs).pooler_output.float().numpy()

    return embed


class Predictor:
    def __init__(self, model_path: str | Path, embed: Callable[[Image.Image], np.ndarray] | None = None):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.threshold = float(bundle["threshold"])
        self.embed = embed or dinov2_embedder()

    def __call__(self, image: Image.Image) -> Result:
        image = image.convert("RGB")
        issue = quality_issue(image)
        if issue is not None:
            return Result("uncertain", None, self.threshold, issue)
        probability = float(self.model.predict_proba(self.embed(image))[0, 1])
        return Result(decide(probability, self.threshold), probability, self.threshold)
