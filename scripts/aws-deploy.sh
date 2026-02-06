#!/bin/bash
# AWS EC2 Deployment Script for Robustness Analysis
# Instance recommendation: c5.18xlarge (72 vCPUs, ~$3.06/hr on-demand)
# Expected runtime: ~7-8 hours for 10,000 permutations
# Estimated cost: ~$21-25

set -e

echo "=========================================="
echo "Book of Mormon Robustness Analysis - AWS Setup"
echo "=========================================="

# Update system
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv git

# Clone repository (or upload your code)
# Option 1: If repo is public or you have SSH keys configured
# git clone https://github.com/YOUR_USERNAME/book-of-mormon-textual-analysis.git
# cd book-of-mormon-textual-analysis

# Option 2: If you uploaded a zip file
# unzip book-of-mormon-textual-analysis.zip
# cd book-of-mormon-textual-analysis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install numpy scipy scikit-learn joblib

# Verify setup
python -c "import numpy; import sklearn; import joblib; print('Dependencies OK')"

echo ""
echo "Setup complete! Run the analysis with:"
echo "  source venv/bin/activate"
echo "  nohup python scripts/run_robustness_optimized.py --permutations 10000 > robustness.log 2>&1 &"
echo ""
echo "Monitor progress with:"
echo "  tail -f robustness.log"
echo ""
echo "Or use screen/tmux for persistent sessions."
