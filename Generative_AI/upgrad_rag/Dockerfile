# Hugging Face Spaces requires port 7860
# Same Dockerfile works for both local and HF deployment
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (Docker caches this layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

# Port 7860 — required by Hugging Face Spaces
EXPOSE 7860

# Start FastAPI on port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
