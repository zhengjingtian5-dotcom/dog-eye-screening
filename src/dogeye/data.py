"""Download the Roboflow "dog-eyes" projects and build an image manifest.

Each Roboflow project holds one condition and is annotated for object
detection (a box around the eye). We crop each image to its largest box so
the model sees the eye rather than the whole dog, and record one row per
crop in a manifest CSV.

Usage:
    ROBOFLOW_API_KEY=... python -m dogeye.data --out data
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

CROP_MARGIN = 0.15
MANIFEST_COLUMNS = ["path", "project", "condition", "label", "source_split", "source_file"]


def normalize_name(name: str) -> str:
    return re.sub(r"[\s_.]+", "-", name.strip().lower())


def load_label_config(path: str | Path) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    config["projects"] = {normalize_name(k): v for k, v in config["projects"].items()}
    return config


def label_for(names: list[str], mapping: dict[str, str]) -> str | None:
    """Return the configured label for the first matching name, or None."""
    for name in names:
        key = normalize_name(name)
        if key in mapping:
            return mapping[key]
    return None


def crop_box(width: int, height: int, bbox: list[float], margin: float = CROP_MARGIN) -> tuple[int, int, int, int]:
    """Expand a COCO [x, y, w, h] box by a margin and clamp it to the image."""
    x, y, w, h = bbox
    dx, dy = w * margin, h * margin
    left = max(0, int(x - dx))
    top = max(0, int(y - dy))
    right = min(width, int(x + w + dx))
    bottom = min(height, int(y + h + dy))
    return left, top, right, bottom


def crops_from_coco(split_dir: Path, out_dir: Path) -> list[dict]:
    """Crop every image in one COCO export folder to its largest box."""
    annotation_file = split_dir / "_annotations.coco.json"
    if not annotation_file.exists():
        return []
    coco = json.loads(annotation_file.read_text())

    largest: dict[int, list[float]] = {}
    for ann in coco.get("annotations", []):
        bbox = ann["bbox"]
        current = largest.get(ann["image_id"])
        if current is None or bbox[2] * bbox[3] > current[2] * current[3]:
            largest[ann["image_id"]] = bbox

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for image in coco["images"]:
        source = split_dir / image["file_name"]
        if not source.exists():
            continue
        with Image.open(source) as img:
            img = img.convert("RGB")
            bbox = largest.get(image["id"])
            if bbox is not None and bbox[2] > 1 and bbox[3] > 1:
                img = img.crop(crop_box(img.width, img.height, bbox))
            target = out_dir / f"{split_dir.name}_{Path(image['file_name']).stem}.jpg"
            img.save(target, quality=95)
        rows.append({"path": str(target), "source_split": split_dir.name, "source_file": image["file_name"]})
    return rows


def latest_version_number(project) -> int:
    return max(int(str(v.version).rsplit("/", 1)[-1]) for v in project.versions())


def download_projects(api_key: str, config: dict, raw_dir: Path) -> list[tuple[str, str, Path]]:
    """Download every mapped project; returns (slug, display name, folder) tuples."""
    from roboflow import Roboflow

    workspace = Roboflow(api_key=api_key).workspace(config["workspace"])
    downloaded = []
    for project_id in workspace.projects():
        slug = project_id.rsplit("/", 1)[-1]
        project = workspace.project(slug)
        label = label_for([slug, project.name], config["projects"])
        if label is None:
            print(f"skip {slug!r}: not in configs/labels.yaml")
            continue
        if label == "exclude":
            print(f"skip {slug!r}: excluded in configs/labels.yaml")
            continue
        location = raw_dir / slug
        version = project.version(latest_version_number(project))
        version.download("coco", location=str(location), overwrite=False)
        downloaded.append((slug, project.name, location))
    return downloaded


def build_manifest(projects: list[tuple[str, str, Path]], config: dict, crop_dir: Path) -> pd.DataFrame:
    rows = []
    for slug, name, location in projects:
        label = label_for([slug, name], config["projects"])
        for split_dir in sorted(p for p in location.iterdir() if p.is_dir()):
            for row in crops_from_coco(split_dir, crop_dir / slug):
                rows.append({**row, "project": slug, "condition": normalize_name(name), "label": label})
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data")
    parser.add_argument("--labels", default="configs/labels.yaml")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY (free at https://app.roboflow.com/settings/api).")

    out = Path(args.out)
    config = load_label_config(args.labels)
    projects = download_projects(api_key, config, out / "raw")
    manifest = build_manifest(projects, config, out / "crops")
    manifest.to_csv(out / "manifest.csv", index=False)
    print(manifest.groupby(["condition", "label"]).size().to_string())
    print(f"wrote {len(manifest)} rows to {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
