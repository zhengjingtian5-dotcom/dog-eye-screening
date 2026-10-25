import io

import joblib
import numpy as np
import pytest
from PIL import Image
from sklearn.linear_model import LogisticRegression

from dogeye.predict import Predictor, decide, quality_issue


def sharp_image(colour=(120, 90, 60)):
    rng = np.random.default_rng(0)
    pixels = rng.integers(60, 200, (64, 64, 3), dtype=np.uint8)
    pixels[20:44, 20:44] = colour
    return Image.fromarray(pixels)


@pytest.fixture
def model_path(tmp_path):
    # Feature 0 > 0 means see_vet.
    X = np.array([[-2.0, 0], [-1.0, 0], [1.0, 0], [2.0, 0]])
    model = LogisticRegression().fit(X, [0, 0, 1, 1])
    path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "threshold": 0.5, "C": 1.0}, path)
    return path


def test_decide_has_an_uncertain_band():
    assert decide(0.9, 0.5) == "see_vet"
    assert decide(0.5, 0.5) == "see_vet"
    assert decide(0.4, 0.5) == "uncertain"
    assert decide(0.2, 0.5) == "normal"


def test_quality_issue_flags_dark_bright_and_blurry():
    assert quality_issue(Image.new("RGB", (64, 64), (5, 5, 5))) == "dark"
    assert quality_issue(Image.new("RGB", (64, 64), (250, 250, 250))) == "bright"
    assert quality_issue(Image.new("RGB", (64, 64), (120, 120, 120))) == "blurry"
    assert quality_issue(sharp_image()) is None


def test_predictor_returns_outcomes(model_path):
    sick = Predictor(model_path, embed=lambda image: np.array([[3.0, 0]]))
    healthy = Predictor(model_path, embed=lambda image: np.array([[-3.0, 0]]))
    assert sick(sharp_image()).outcome == "see_vet"
    assert healthy(sharp_image()).outcome == "normal"

    poor = sick(Image.new("RGB", (64, 64), (5, 5, 5)))
    assert poor.outcome == "uncertain" and poor.quality_issue == "dark" and poor.probability is None


def test_api_and_page(model_path, monkeypatch):
    from fastapi.testclient import TestClient

    import app as demo_app

    predictor = Predictor(model_path, embed=lambda image: np.array([[3.0, 0]]))
    monkeypatch.setattr(demo_app, "predictor", lambda: predictor)
    client = TestClient(demo_app.app)

    buffer = io.BytesIO()
    sharp_image().save(buffer, format="PNG")
    response = client.post("/api/screen", files={"file": ("eye.png", buffer.getvalue(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "see_vet"
    assert "not a diagnosis" in body["disclaimer"]

    assert client.get("/").status_code == 200
    assert "看兽医" in demo_app.screen(sharp_image(), "zh")
    assert "see a vet" in demo_app.screen(sharp_image(), "en")
