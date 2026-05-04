@echo off
cd /d %~dp0
if not exist model_artifacts\trained_extra_trees.pkl (
  echo [INFO] Chua co model da train. Dang train truoc...
  python train_model.py
)
streamlit run app.py
