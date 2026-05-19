#!/bin/bash

set -e

echo "🚀 Starting training pipeline on VM"

# =========================
# 1. System dependencies
# =========================
# sudo apt update
# sudo apt install -y git python3-venv python3-pip curl

# =========================
# 2. Install Poetry
# =========================
# curl -sSL https://install.python-poetry.org | python3 -
# export PATH="$HOME/.local/bin:$PATH"

# poetry --version

# =========================
# 3. Clone repo
# =========================
# if [ ! -d "multi-index-rag" ]; then
#   git clone --branch feature/formula-without-latexml https://github.com/abramjopaul/multi-index-rag.git
# fi

# cd multi-index-rag

# =========================
# 4. Poetry setup
# =========================
# poetry config virtualenvs.in-project true
# poetry env use python3
# poetry install

# =========================
# 5. Dataset download
# =========================
BUCKET_NAME="multi-index-rag-bucket"

mkdir -p data/raw/collection/formula/opt_representation_v3

gsutil -m cp -r \
gs://$BUCKET_NAME/data/raw/collection/formula/opt_representation_v3/* \
data/raw/collection/formula/opt_representation_v3/

echo "✓ Dataset ready"

# =========================
# 6. Training
# =========================
poetry run python experiments/train_formula_models.py \
  -t OPT \
  -w 12

# =========================
# 7. Upload model
# =========================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

gsutil -m cp -r \
data/formula-indexing/opt/* \
gs://$BUCKET_NAME/models/opt_$TIMESTAMP/

echo "🎉 Training complete"