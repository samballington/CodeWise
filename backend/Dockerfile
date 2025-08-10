FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including Node.js for Mermaid validation
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Mermaid CLI globally for command line usage
RUN npm install -g @mermaid-js/mermaid-cli

# Copy production requirements first for better caching
COPY requirements-prod.txt ./requirements-prod.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

# Copy source into /app/backend so imports use backend.*
COPY . /app/backend

# Install Node.js dependencies for mermaid validation
WORKDIR /app/backend
RUN npm install

# Environment variables
ENV WORKSPACE_DIR=/workspace
ENV OPENAI_API_KEY=""
ENV PYTHONPATH=/app:/app/backend

# Set working directory to backend for uvicorn
WORKDIR /app/backend

# Start the FastAPI server directly
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

EXPOSE 8000 