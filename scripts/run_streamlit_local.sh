#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 -m streamlit run app/streamlit_app.py "$@"
