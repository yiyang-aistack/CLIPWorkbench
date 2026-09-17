FROM python:3.11-slim

# System dependencies for PIL/OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[onnx]"

# Copy source code
COPY src/ src/
COPY configs/ configs/

EXPOSE 8000

CMD ["uvicorn", "clipworkbench.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
