FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create and configure entrypoint script
RUN echo '#!/bin/bash\nset -e\n\n# Start the FastAPI server\nexec uvicorn main:app --host 0.0.0.0 --port 8000' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Use entrypoint to start server
ENTRYPOINT ["/entrypoint.sh"]

# Expose port
EXPOSE 8000 