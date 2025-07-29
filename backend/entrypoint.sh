#!/bin/bash
set -e

if [ "$RUN_MODE" = "test" ]; then
  # Run only pytest then exit
  shift || true
  pytest -q "$@"
  exit $?
fi

# Default behaviour: start FastAPI server
exec uvicorn main:app --host 0.0.0.0 --port 8000 