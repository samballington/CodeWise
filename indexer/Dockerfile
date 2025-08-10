FROM python:3.11-slim

WORKDIR /app

# Install production deps first for better caching
COPY requirements-prod.txt ./requirements-prod.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

# Copy source code into /app/indexer
COPY . /app/indexer

ENV WORKSPACE_DIR=/workspace
ENV OPENAI_API_KEY=""
ENV PYTHONPATH=/app/indexer

CMD ["uvicorn", "indexer.api:app", "--host", "0.0.0.0", "--port", "8002"] 