import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from dogeye.data import build_manifest, crop_box, label_for, load_label_config, normalize_name
from dogeye.groups import (
    close_hash_pairs,
    duplicate_groups,
    group_split,
    same_source_pairs,
    similar_embedding_pairs,
    source_stem,
)
from dogeye.metrics import binary_report, expected_calibration_error, threshold_for_recall


def test_normalize_name_matches_config_keys():
    assert normalize_name("Nuclear sclerosis") == "nuclear-sclerosis"
    assert normalize_name("DOG-EYES_ver.4") == "dog-eyes-ver-4"


def test_label_config_maps_display_names():
    config = load_label_config("configs/labels.yaml")
    assert label_for(["eyelid tumor"], config["projects"]) == "see_vet"
    assert label_for(["healthy"], config["projects"]) == "normal"
    assert label_for(["DOG-EYES_ver.4"], config["projects"]) == "exclude"
    assert label_for(["unknown-project"], config["projects"]) is None


def test_crop_box_adds_margin_and_clamps():
    assert crop_box(100, 100, [10, 10, 20, 20], margin=0.5) == (0, 0, 40, 40)
    assert crop_box(100, 100, [90, 90, 20, 20], margin=0.0) == (90, 90, 100, 100)


def write_coco_split(split_dir, n, colour, seed):
    split_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    images, annotations = [], []
    for i in range(n):
        pixels = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        pixels[16:48, 16:48] = colour
        Image.fromarray(pixels).save(split_dir / f"img{i}.jpg")
        images.append({"id": i, "file_name": f"img{i}.jpg", "width": 64, "height": 64})
        annotations.append({"id": i, "image_id": i, "bbox": [16, 16, 32, 32]})
    (split_dir / "_annotations.coco.json").write_text(json.dumps({"images": images, "annotations": annotations}))


def test_build_manifest_crops_and_labels(tmp_path):
    config = load_label_config("configs/labels.yaml")
    write_coco_split(tmp_path / "raw/healthy/train", 3, (200, 50, 50), seed=0)
    write_coco_split(tmp_path / "raw/conjunctivitis/valid", 2, (50, 50, 200), seed=1)
    projects = [("healthy", "healthy", tmp_path / "raw/healthy"),
                ("conjunctivitis", "conjunctivitis", tmp_path / "raw/conjunctivitis")]

    manifest = build_manifest(projects, config, tmp_path / "crops")

    assert len(manifest) == 5
    assert manifest.groupby("label").size().to_dict() == {"normal": 3, "see_vet": 2}
    with Image.open(manifest["path"].iloc[0]) as img:
        assert img.size == (41, 41)  # 32px box plus 15% margin each side


def test_near_duplicates_are_grouped():
    hashes = np.array([0b0, 0b1, 0xFFFF_FFFF_FFFF_FFFF], dtype=np.uint64)
    features = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0001]])
    groups = duplicate_groups(3, [close_hash_pairs(hashes), similar_embedding_pairs(features)])
    # 0~1 by hash, 1~2 by embedding: one transitive group.
    assert len(set(groups)) == 1


def test_roboflow_augmentations_share_a_source():
    a = "D21_3daf27f0-60a5_jpg.rf.4f00de54be173a31235502be092aabb8.jpg"
    b = "D21_3daf27f0-60a5_jpg.rf.004a9f37b8273c5ceee2694b926e4776.jpg"
    assert source_stem(a) == source_stem(b) == "D21_3daf27f0-60a5"
    projects = pd.Series(["healthy", "healthy", "entropion"])
    files = pd.Series([a, b, a])
    # Same upload name in another project is a different photo.
    assert list(same_source_pairs(projects, files)) == [(0, 1)]
    assert len(set(duplicate_groups(3, [same_source_pairs(projects, files)]))) == 2


def test_group_split_never_leaks_a_group():
    groups = np.repeat(np.arange(200), 3)
    labels = pd.Series(np.where(groups % 3 == 0, "see_vet", "normal"))
    split = group_split(labels, groups, seed=0)

    per_group = pd.DataFrame({"group": groups, "split": split}).groupby("group")["split"].nunique()
    assert (per_group == 1).all()
    shares = pd.Series(split).value_counts(normalize=True)
    assert 0.6 < shares["train"] < 0.8
    for name in ["val", "test"]:
        assert set(labels[split == name]) == {"normal", "see_vet"}


def test_groups_cli_folds_copied_images(tmp_path):
    config = load_label_config("configs/labels.yaml")
    write_coco_split(tmp_path / "raw/healthy/train", 20, (200, 50, 50), seed=0)
    write_coco_split(tmp_path / "raw/conjunctivitis/train", 20, (50, 50, 200), seed=1)
    projects = [("healthy", "healthy", tmp_path / "raw/healthy"),
                ("conjunctivitis", "conjunctivitis", tmp_path / "raw/conjunctivitis")]
    manifest = build_manifest(projects, config, tmp_path / "crops")
    # Simulate a re-uploaded copy of the first healthy image.
    copy = manifest.iloc[[0]].copy()
    manifest = pd.concat([manifest, copy], ignore_index=True)
    manifest.to_csv(tmp_path / "manifest.csv", index=False)
    features = np.random.default_rng(0).normal(size=(len(manifest), 8))
    features[-1] = features[0]
    np.savez_compressed(tmp_path / "features.npz", features=features)

    subprocess.run(
        [sys.executable, "-m", "dogeye.groups", "--manifest", str(tmp_path / "manifest.csv"),
         "--features", str(tmp_path / "features.npz"), "--out", str(tmp_path / "split.csv")],
        check=True,
    )

    result = pd.read_csv(tmp_path / "split.csv")
    assert result["group"].nunique() == 40
    assert result.loc[0, "split"] == result.loc[40, "split"]
    assert result["feature_row"].tolist() == list(range(41))


def test_threshold_for_recall_hits_target():
    y = np.array([1, 1, 1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.4, 0.2, 0.3, 0.1])
    t = threshold_for_recall(y, scores, 0.75)
    assert t == 0.4
    assert binary_report(y, scores, t)["recall"] == 0.75


def test_ece_is_zero_when_calibrated_and_large_when_not():
    assert expected_calibration_error(np.array([0, 1]), np.array([0.0, 1.0])) == 0.0
    assert expected_calibration_error(np.array([0, 0]), np.array([0.95, 0.95])) == pytest.approx(0.95)


def test_train_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    n = 600
    groups = np.repeat(np.arange(n // 2), 2)
    y = (groups % 2).astype(int)
    features = rng.normal(size=(n, 16)) + y[:, None] * 1.5
    manifest = pd.DataFrame({
        "path": [f"img{i}.jpg" for i in range(n)],
        "condition": np.where(y == 1, "conjunctivitis", "healthy"),
        "label": np.where(y == 1, "see_vet", "normal"),
        "group": groups,
        "feature_row": np.arange(n),
    })
    manifest["split"] = group_split(manifest["label"], groups, seed=0)
    manifest.to_csv(tmp_path / "manifest_split.csv", index=False)
    np.savez_compressed(tmp_path / "features.npz", features=features)

    subprocess.run(
        [sys.executable, "-m", "dogeye.train", "--manifest", str(tmp_path / "manifest_split.csv"),
         "--features", str(tmp_path / "features.npz"), "--out", str(tmp_path / "run")],
        check=True,
    )

    report = json.loads((tmp_path / "run/metrics.json").read_text())
    assert report["val"]["recall"] >= 0.95
    assert report["test"]["auroc"] > 0.9
    for name in ["model.joblib", "per_condition.csv", "confusion_matrix.png", "reliability.png"]:
        assert (tmp_path / "run" / name).exists()
