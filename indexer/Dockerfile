FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV WORKSPACE_DIR=/workspace
ENV OPENAI_API_KEY=""

CMD ["python", "main.py"] 