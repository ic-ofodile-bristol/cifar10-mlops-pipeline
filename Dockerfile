
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_URI=/app/model

WORKDIR /app

# Install CPU-only PyTorch and Torchvision
RUN pip install torch==2.15.0 torchvision==0.29.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Install serving dependencies
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# Copy application and model
COPY model/ model/
COPY src/ src/
COPY app.py .

# Run application as a non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
