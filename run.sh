#!/usr/bin/env bash
# Run the War Predictive Streamlit Dashboard
# Usage: bash run.sh

cd "$(dirname "$0")"

# Install dependencies if needed
pip install -r requirements.txt

# Launch Streamlit
streamlit run app/streamlit_app.py
