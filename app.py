"""Bilingual (English / 中文) demo: a Gradio page and a JSON API on FastAPI.

    pip install -e ".[features,app]"
    MODEL_PATH=models/baseline.joblib uvicorn app:app --port 7860

Open http://localhost:7860 for the page, or POST a photo to /api/screen.
"""

from __future__ import annotations

import io
import os
from functools import lru_cache

import gradio as gr
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from dogeye.predict import Predictor, Result

MODEL_PATH = os.environ.get("MODEL_PATH", "models/baseline.joblib")

TEXT = {
    "en": {
        "title": "Dog eye screening",
        "intro": "Upload a close-up photo of **one** dog eye. Crop it so the eye fills most of the picture.",
        "disclaimer": "This is a portfolio demo, **not a diagnosis**. It can miss problems. "
        "If you are worried about your dog's eyes, see a vet.",
        "image": "Eye photo",
        "button": "Check",
        "see_vet": "### Possible problem: please see a vet\nThe photo shows signs that are worth a vet's look.",
        "uncertain": "### Not sure: retake the photo or see a vet\nThe model could not decide confidently.",
        "normal": "### Looks normal\nNo visible sign that needs a vet in this photo. Keep watching for redness, "
        "discharge, squinting or cloudiness.",
        "dark": "The photo is too dark.",
        "bright": "The photo is too bright.",
        "blurry": "The photo is too blurry.",
        "score": "Model score {p:.2f} (flag threshold {t:.2f})",
        "language": "Language / 语言",
    },
    "zh": {
        "title": "狗狗眼睛健康筛查",
        "intro": "请上传一张狗狗**单只眼睛**的近照，并裁剪到眼睛占画面的大部分。",
        "disclaimer": "这是作品集演示，**不是诊断**，可能会漏掉问题。如果担心狗狗的眼睛，请带它去看兽医。",
        "image": "眼睛照片",
        "button": "开始检查",
        "see_vet": "### 可能有问题：建议看兽医\n照片里有值得兽医检查的迹象。",
        "uncertain": "### 无法确定：请重拍或去看兽医\n模型对这张照片没有把握。",
        "normal": "### 看起来正常\n这张照片里没有发现需要看兽医的迹象。请继续留意发红、分泌物、眯眼或混浊。",
        "dark": "照片太暗。",
        "bright": "照片太亮。",
        "blurry": "照片太模糊。",
        "score": "模型分数 {p:.2f}（提示就医阈值 {t:.2f}）",
        "language": "Language / 语言",
    },
}


@lru_cache(maxsize=1)
def predictor() -> Predictor:
    return Predictor(MODEL_PATH)


def to_json(result: Result) -> dict:
    return {
        "outcome": result.outcome,
        "probability": result.probability,
        "threshold": result.threshold,
        "quality_issue": result.quality_issue,
        "disclaimer": TEXT["en"]["disclaimer"].replace("**", ""),
    }


def render(result: Result, lang: str) -> str:
    text = TEXT[lang]
    parts = [text[result.outcome]]
    if result.quality_issue:
        parts.append(text[result.quality_issue])
    if result.probability is not None:
        parts.append(text["score"].format(p=result.probability, t=result.threshold))
    parts.append(f"> {text['disclaimer']}")
    return "\n\n".join(parts)


def screen(image: Image.Image | None, lang: str) -> str:
    if image is None:
        return TEXT[lang]["intro"]
    return render(predictor()(image), lang)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Dog eye screening") as demo:
        lang = gr.Radio([("English", "en"), ("中文", "zh")], value="zh", label=TEXT["zh"]["language"])
        title = gr.Markdown(f"# {TEXT['zh']['title']}\n\n{TEXT['zh']['intro']}\n\n> {TEXT['zh']['disclaimer']}")
        image = gr.Image(type="pil", label=TEXT["zh"]["image"], sources=["upload", "webcam", "clipboard"])
        button = gr.Button(TEXT["zh"]["button"], variant="primary")
        output = gr.Markdown()

        def switch(lang_value):
            t = TEXT[lang_value]
            return (
                f"# {t['title']}\n\n{t['intro']}\n\n> {t['disclaimer']}",
                gr.update(label=t["image"]),
                gr.update(value=t["button"]),
                "",
            )

        lang.change(switch, lang, [title, image, button, output])
        button.click(screen, [image, lang], output)
    return demo


app = FastAPI(title="Dog eye screening API")


@app.post("/api/screen")
async def screen_api(file: UploadFile = File(...)) -> dict:
    image = Image.open(io.BytesIO(await file.read()))
    return to_json(predictor()(image))


app = gr.mount_gradio_app(app, build_demo(), path="/")
