#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -d "$PROJECT_DIR/.venv" ]; then
    VENV_PATH="$PROJECT_DIR/.venv"
else
    VENV_PATH="${VENV_PATH:-$HOME/Documents/.venv}"
fi

if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment tidak ditemukan: $VENV_PATH"
    echo "Buat terlebih dahulu dengan:"
    echo "python3 -m venv --system-site-packages $VENV_PATH"
    exit 1
fi

source "$VENV_PATH/bin/activate"
cd "$PROJECT_DIR"

exec python app.py
