#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Lightning AI L40S Studio — Environment Setup Script
#  Run this ONCE after creating a new studio with L40S GPU
#  Usage: bash setup_env.sh
# ═══════════════════════════════════════════════════════════════

set -e  # exit on any error

echo "========================================================"
echo "  Lightning AI L40S — Environment Setup"
echo "========================================================"

# ── 1. System info ───────────────────────────────────────────
echo ""
echo "[1/7] System Info"
echo "-----------------"
nvidia-smi
echo ""
python3 --version
pip --version

# ── 2. Upgrade pip ───────────────────────────────────────────
echo ""
echo "[2/7] Upgrading pip ..."
pip install --upgrade pip --quiet

# ── 3. Install PyTorch with CUDA 12.1 (matches L40S) ────────
echo ""
echo "[3/7] Installing PyTorch (CUDA 12.1) ..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    --quiet

# Verify GPU is visible
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA NOT available!'
print(f'  PyTorch  : {torch.__version__}')
print(f'  CUDA     : {torch.version.cuda}')
print(f'  GPU      : {torch.cuda.get_device_name(0)}')
print(f'  VRAM     : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
print('  PyTorch + CUDA OK')
"

# ── 4. Install FAISS GPU ─────────────────────────────────────
echo ""
echo "[4/7] Installing FAISS GPU ..."
pip install faiss-gpu --quiet
python3 -c "import faiss; print(f'  faiss version : {faiss.__version__}  OK')"

# ── 5. Install HuggingFace + other deps ──────────────────────
echo ""
echo "[5/7] Installing transformers, pandas, tqdm, ir-datasets ..."
pip install \
    transformers==4.40.0 \
    accelerate \
    pandas \
    tqdm \
    ir-datasets \
    numpy \
    --quiet

python3 -c "
import transformers, pandas, tqdm, numpy, ir_datasets
print(f'  transformers : {transformers.__version__}')
print(f'  pandas       : {pandas.__version__}')
print(f'  numpy        : {numpy.__version__}')
print('  All deps OK')
"

# ── 6. Create directory structure ───────────────────────────
echo ""
echo "[6/7] Creating directories ..."
STUDIO="/teamspace/studios/this_studio"

mkdir -p "$STUDIO/data/e5_index/checkpoints"
mkdir -p "$STUDIO/logs"
mkdir -p "$STUDIO/data"

echo "  $STUDIO/data/e5_index/checkpoints  ✅"
echo "  $STUDIO/logs                        ✅"
echo "  $STUDIO/data                        ✅"

# ── 7. Download IndicMARCO dataset via ir_datasets ───────────
echo ""
echo "[7/7] Pre-downloading IndicMARCO Hindi via ir_datasets ..."
echo "  (This will take 5-10 min first time — cached after)"
python3 -c "
import ir_datasets
print('  Downloading mmarco/v2/hi/train ...')
ds = ir_datasets.load('mmarco/v2/hi/train')
count = sum(1 for _ in ds.docs_iter())
print(f'  Passages : {count:,}')
print('  Dataset ready.')
"

echo ""
echo "========================================================"
echo "  SETUP COMPLETE"
echo "========================================================"
echo ""
echo "  Now run the indexing pipeline:"
echo "  nohup python build_index_l40s.py > logs/run.log 2>&1 &"
echo "  tail -f logs/run.log"
echo ""
