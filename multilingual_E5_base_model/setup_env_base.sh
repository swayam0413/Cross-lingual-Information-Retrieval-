#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Lightning AI L40S Studio — Environment Setup for e5-BASE
#  Run ONCE after creating a new studio with L40S GPU
#  Usage: bash setup_env_base.sh
# ═══════════════════════════════════════════════════════════════

set -e  # exit immediately on any error

echo "========================================================"
echo "  Lightning AI L40S — e5-BASE Environment Setup"
echo "========================================================"

# ── 1. System info ───────────────────────────────────────────
echo ""
echo "[1/8] System Info"
echo "-----------------"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""
python3 --version
pip --version
echo ""
free -h
df -h /teamspace 2>/dev/null || df -h /

# ── 2. Fix numpy FIRST (must be before everything else) ──────
echo ""
echo "[2/8] Installing compatible numpy (<2) ..."
pip install "numpy<2" --force-reinstall --quiet
python3 -c "
import numpy as np
print(f'  numpy : {np.__version__}')
assert int(np.__version__.split('.')[0]) < 2, 'numpy must be < 2.0'
print('  numpy OK')
"

# ── 3. Install PyTorch with CUDA 12.1 ────────────────────────
echo ""
echo "[3/8] Installing PyTorch 2.x (CUDA 12.1) ..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    --quiet

python3 -c "
import torch
assert torch.cuda.is_available(), 'ERROR: CUDA not available — check GPU is L40S'
print(f'  PyTorch  : {torch.__version__}')
print(f'  CUDA     : {torch.version.cuda}')
print(f'  GPU      : {torch.cuda.get_device_name(0)}')
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'  VRAM     : {vram:.1f} GB')
if vram < 40:
    print(f'  WARNING: Expected ~48GB for L40S, got {vram:.1f}GB')
else:
    print('  VRAM OK for e5-base')
"

# ── 4. Install FAISS GPU ─────────────────────────────────────
echo ""
echo "[4/8] Installing FAISS GPU ..."
pip install faiss-gpu --quiet \
    || pip install faiss-gpu-cu12 --quiet \
    || { echo "  Falling back to faiss-cpu"; pip install faiss-cpu --quiet; }

python3 -c "
import faiss
print(f'  faiss : {faiss.__version__}  OK')
"

# ── 5. Install all Python dependencies ───────────────────────
echo ""
echo "[5/8] Installing transformers, pandas, tqdm, ir-datasets, scipy ..."
pip install \
    "transformers>=4.40.0" \
    "accelerate>=0.27.0" \
    pandas \
    tqdm \
    "ir-datasets>=0.5.5" \
    scipy \
    scikit-learn \
    --quiet

# Reinstall scipy + sklearn against the pinned numpy<2
echo "  Reinstalling scipy + sklearn against numpy<2 ..."
pip install scipy scikit-learn --force-reinstall --quiet

# Final verification
python3 -c "
import transformers, pandas, tqdm, numpy, ir_datasets, scipy, sklearn, torch, faiss
print(f'  transformers : {transformers.__version__}')
print(f'  pandas       : {pandas.__version__}')
print(f'  numpy        : {numpy.__version__}')
print(f'  scipy        : {scipy.__version__}')
print(f'  sklearn      : {sklearn.__version__}')
print(f'  torch        : {torch.__version__}')
print(f'  faiss        : {faiss.__version__}')
print('  All deps OK')
"

# ── 6. Pre-download e5-base model weights ────────────────────
echo ""
echo "[6/8] Pre-downloading multilingual-e5-base model weights ..."
echo "  (~1.1GB download — cached to ~/.cache/huggingface)"
python3 -c "
from transformers import AutoTokenizer, AutoModel
import torch

MODEL = 'intfloat/multilingual-e5-base'
print(f'  Downloading {MODEL} ...')
tok = AutoTokenizer.from_pretrained(MODEL)
m   = AutoModel.from_pretrained(MODEL)
n_params = sum(p.numel() for p in m.parameters()) / 1e6
print(f'  Params   : {n_params:.0f}M')
print(f'  FP16 VRAM: ~{n_params * 2 / 1e3:.0f}MB')

# Verify it works on GPU
if torch.cuda.is_available():
    m = m.half().cuda().eval()
    dummy = tok(['query: test'], return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = m(**dummy)
    print(f'  Output shape: {out.last_hidden_state.shape}')
    print(f'  GPU forward pass OK')
print('  Model download and test OK')
"

# ── 7. Create directory structure ────────────────────────────
echo ""
echo "[7/8] Creating directory structure ..."
STUDIO="/teamspace/studios/this_studio"

mkdir -p "$STUDIO/data/e5base_index/checkpoints"
mkdir -p "$STUDIO/logs"
mkdir -p "$STUDIO/data"

echo "  $STUDIO/data/e5base_index/checkpoints   created"
echo "  $STUDIO/logs                             created"
echo "  $STUDIO/data                             created"

# ── 8. Download IndicMARCO + export queries/qrels ────────────
echo ""
echo "[8/8] Downloading IndicMARCO Hindi dataset + exporting queries/qrels ..."
echo "  Collection download: ~7GB TSV — takes 5-15 min first time"

python3 -c "
import ir_datasets
from pathlib import Path

STUDIO   = Path('/teamspace/studios/this_studio')
DATA_DIR = STUDIO / 'data'

# ── Download collection ──────────────────────────────────────
print('  Downloading mmarco/v2/hi collection ...')
ds = ir_datasets.load('mmarco/v2/hi')
count = 0
for i, doc in enumerate(ds.docs_iter()):
    count += 1
    if i == 0:
        print(f'  First doc id: {doc.doc_id}')
    if (i+1) % 2_000_000 == 0:
        print(f'  Progress: {i+1:,} docs downloaded ...')
print(f'  Collection total: {count:,} passages')

# Verify TSV file
TSV = STUDIO / '.ir_datasets/mmarco/v2/hindi_collection.tsv'
if TSV.exists():
    size_gb = TSV.stat().st_size / 1e9
    print(f'  TSV saved: {TSV}  ({size_gb:.1f}GB)')
else:
    import glob
    found = glob.glob(str(STUDIO / '.ir_datasets/**/*.tsv'), recursive=True)
    if found:
        print(f'  TSV found at: {found[0]}')
    else:
        print('  WARNING: TSV file not found at expected path')

# ── Export queries from dev/small ────────────────────────────
QUERIES_FILE = DATA_DIR / 'trec_dl19_hindi_query.tsv'
print(f'  Exporting queries from mmarco/v2/hi/dev/small ...')
ds_eval = ir_datasets.load('mmarco/v2/hi/dev/small')
n_q = 0
with open(QUERIES_FILE, 'w', encoding='utf-8') as f:
    for q in ds_eval.queries_iter():
        f.write(f'{q.query_id}\t{q.text}\n')
        n_q += 1
print(f'  Queries written: {n_q:,}  -> {QUERIES_FILE}')

# ── Export qrels ─────────────────────────────────────────────
QRELS_FILE = DATA_DIR / 'pass_2019.qrels'
print(f'  Exporting qrels ...')
n_qr = 0
with open(QRELS_FILE, 'w', encoding='utf-8') as f:
    for qr in ds_eval.qrels_iter():
        f.write(f'{qr.query_id} 0 {qr.doc_id} {qr.relevance}\n')
        n_qr += 1
print(f'  Qrels written: {n_qr:,}  -> {QRELS_FILE}')

print('  Dataset export complete.')
"

# ── Final verification ────────────────────────────────────────
echo ""
echo "Verifying all input files ..."
COLLECTION="/teamspace/studios/this_studio/.ir_datasets/mmarco/v2/hindi_collection.tsv"
QUERIES="/teamspace/studios/this_studio/data/trec_dl19_hindi_query.tsv"
QRELS="/teamspace/studios/this_studio/data/pass_2019.qrels"

check_file() {
    if [ -f "$1" ]; then
        SIZE=$(du -sh "$1" | cut -f1)
        LINES=$(wc -l < "$1")
        echo "  OK  $1  ($SIZE, $LINES lines)"
    else
        echo "  MISSING  $1"
    fi
}

check_file "$COLLECTION"
check_file "$QUERIES"
check_file "$QRELS"

echo ""
echo "========================================================"
echo "  SETUP COMPLETE"
echo "========================================================"
echo ""
echo "  Files ready:"
echo "    Collection : $COLLECTION"
echo "    Queries    : $QUERIES"
echo "    Qrels      : $QRELS"
echo ""
echo "  Run the pipeline:"
echo "    python build_index_e5base_l40s.py"
echo ""
echo "  Or background:"
echo "    nohup python build_index_e5base_l40s.py > logs/run_base.log 2>&1 &"
echo "    tail -f logs/run_base.log"
echo ""
echo "  Monitor GPU:"
echo "    watch -n 5 nvidia-smi"
echo ""
echo "  Check progress:"
echo "    cat data/e5base_index/checkpoints/progress.json"
echo ""
echo "  Kill if needed:"
echo "    pkill -f build_index_e5base_l40s.py"
echo ""
