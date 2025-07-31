FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source into /app/backend so imports use backend.*
COPY . /app/backend

# Environment variables
ENV WORKSPACE_DIR=/workspace
ENV OPENAI_API_KEY=""
ENV PYTHONPATH=/app:/app/backend

# Set working directory to backend for uvicorn
WORKDIR /app/backend

# Start the FastAPI server directly
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

EXPOSE 8000 