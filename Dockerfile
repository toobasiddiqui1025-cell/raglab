FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for PDF processing and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications and install
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy application source code
COPY src/ src/
COPY app/ app/
COPY configs/ configs/

EXPOSE 8000 8501

CMD ["uvicorn", "raglab.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
