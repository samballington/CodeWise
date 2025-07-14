#!/bin/bash
set -e

# Run backend tests; exit if they fail
pytest -q

# Start the FastAPI server
exec uvicorn main:app --host 0.0.0.0 --port 8000 