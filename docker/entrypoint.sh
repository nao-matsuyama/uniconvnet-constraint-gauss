#!/bin/bash
set -e

# Activate virtual environment if present
if [ -f "/workspace/.venv/bin/activate" ]; then
  source /workspace/.venv/bin/activate
fi

exec "$@"
