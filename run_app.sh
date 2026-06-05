#!/bin/bash
# Coagulation Quantification — Launch Gradio Web App
cd "$(dirname "$0")"
source .venv/bin/activate
echo "Starting Coagulation Quantification App..."
echo "Open http://127.0.0.1:7860 in your browser"
python3 app.py
