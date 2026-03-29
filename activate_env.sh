#!/bin/bash
# Script to activate the Python virtual environment for the ML Product Pricing project

echo "Activating Python virtual environment..."
source venv/bin/activate

echo "Virtual environment activated!"
echo "Python path: $(which python3)"
echo "Pip path: $(which pip)"

echo ""
echo "Available packages:"
pip list | grep -E "(pandas|numpy|torch|transformers|scikit-learn)" | head -5

echo ""
echo "To deactivate the environment, run: deactivate"
echo "To run Python with the environment: python3 your_script.py"