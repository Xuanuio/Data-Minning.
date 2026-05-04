#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -f "model_artifacts/trained_extra_trees.pkl" ]; then
  echo "[INFO] Chua co model da train. Dang train truoc..."
  python train_model.py
fi
streamlit run app.py
