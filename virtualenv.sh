#!/bin/bash
# This script creates a Python virtual environment and installs dependencies from requirements.txt

python -m venv venv --upgrade-deps
source venv/bin/activate
pip install -r requirements.txt
