FROM python:3.11-slim

WORKDIR /app

# Install deps first for better caching
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code into /app/indexer
COPY . /app/indexer

ENV WORKSPACE_DIR=/workspace
ENV OPENAI_API_KEY=""
ENV PYTHONPATH=/app/indexer

CMD ["uvicorn", "indexer.api:app", "--host", "0.0.0.0", "--port", "8002"] 