# Hugging Face Spaces (Docker SDK) image for the demo.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src ./src
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir ".[features,app]" "scikit-learn==1.6.1"

USER user
ENV HF_HOME=/home/user/.cache/huggingface MODEL_PATH=models/baseline.joblib
# Bake the DINOv2 weights into the image so the first request is fast.
RUN python -c "from transformers import AutoImageProcessor, AutoModel; \
AutoImageProcessor.from_pretrained('facebook/dinov2-small'); AutoModel.from_pretrained('facebook/dinov2-small')"

COPY --chown=user app.py ./
COPY --chown=user models ./models

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
