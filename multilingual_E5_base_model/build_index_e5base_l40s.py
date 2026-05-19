"""
═══════════════════════════════════════════════════════════════
  Dense Index Builder — multilingual-E5-BASE on L40S 48GB
  Model   : intfloat/multilingual-e5-base
  Dataset : IndicMARCO Hindi (direct TSV read)
  Platform: Lightning AI Studio (L40S — 48GB VRAM, 16 CPUs)
═══════════════════════════════════════════════════════════════
  KEY DIFFERENCES vs e5-small:
  - EMBED_DIM  : 384  →  768  (2x larger embeddings)
  - Model size : ~117M params  →  ~278M params
  - VRAM FP16  : ~234MB  →  ~556MB
  - mmap size  : ~13GB  →  ~26GB on disk
  - Batch size : auto-probed (will be smaller than e5-small)

  WHY IT WON'T GET KILLED:
  - mmap writes embeddings directly to disk (zero RAM accumulation)
  - FAISS built in 500K chunks (never loads 26GB into RAM at once)
  - Checkpoint = flush mmap + save progress JSON (no np.vstack)
  - Auto-resume: reopens mmap at exact passage offset on restart

  HOW TO RUN:
      python build_index_e5base_l40s.py
      # background:
      nohup python build_index_e5base_l40s.py > logs/run_base.log 2>&1 &
      tail -f logs/run_base.log
      # check progress:
      cat data/e5base_index/checkpoints/progress.json
═══════════════════════════════════════════════════════════════
"""

# ── CUDA memory config BEFORE torch import ───────────────────
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"

import gc
import json
import logging
import math
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

STUDIO_DIR = Path("/teamspace/studios/this_studio")
DATA_DIR   = STUDIO_DIR / "data"

# ── Input files ──────────────────────────────────────────────
COLLECTION_TSV = Path("/teamspace/studios/this_studio/.ir_datasets/mmarco/v2/hindi_collection.tsv")
QUERIES_FILE   = DATA_DIR / "trec_dl19_hindi_query.tsv"
QRELS_FILE     = DATA_DIR / "pass_2019.qrels"

# ── Output dirs — separate from e5-small to avoid conflicts ──
OUTPUT_DIR = DATA_DIR / "e5base_index"
LOG_DIR    = STUDIO_DIR / "logs"
CKPT_DIR   = OUTPUT_DIR / "checkpoints"

for d in [OUTPUT_DIR, LOG_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Output files ─────────────────────────────────────────────
INDEX_PATH     = OUTPUT_DIR / "index.faiss"
DOCID_MAP_PATH = OUTPUT_DIR / "docid_map.json"
RUN_FILE       = OUTPUT_DIR / "retrieval_run_top100.trec"
RESULTS_FILE   = OUTPUT_DIR / "evaluation_results.json"

# ── Checkpoint files ─────────────────────────────────────────
MMAP_PATH     = CKPT_DIR / "embeddings.mmap"
DOCID_CKPT    = CKPT_DIR / "docids.json"
PROGRESS_CKPT = CKPT_DIR / "progress.json"

# ── Log file ─────────────────────────────────────────────────
LOG_FILE = LOG_DIR / f"run_base_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# ── Model — only change from e5-small ────────────────────────
MODEL_NAME = "intfloat/multilingual-e5-base"
EMBED_DIM  = 768    # e5-base output dim (was 384 for e5-small)

# ── L40S 48GB Hyperparams for e5-BASE ────────────────────────
#   e5-base FP16 model ~556MB VRAM
#   Batch 4096 × 256 tokens × FP16 × 768-dim ≈ ~18-20GB VRAM
#   → safe on L40S 48GB with 4GB headroom at 44GB limit
BATCH_SIZE       = 4096    # starting point — auto-probed downward if needed
MAX_LENGTH       = 256     # 256 sufficient for Hindi passages; 2x faster than 512
TOP_K            = 100
EVAL_K           = 10
MAX_PASSAGES     = None    # None = full 8.8M dataset
CHECKPOINT_EVERY = 50_000  # flush mmap + save docids every 50K passages
VRAM_LIMIT_GB    = 44.0    # use 44 of 48GB, keep 4GB headroom


# ═══════════════════════════════════════════════════════════════
#  LOGGING — terminal + timestamped log file
# ═══════════════════════════════════════════════════════════════

def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("e5base_l40s")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

log = setup_logger(LOG_FILE)


# ═══════════════════════════════════════════════════════════════
#  GRACEFUL SHUTDOWN — checkpoint on Ctrl+C or SIGTERM
# ═══════════════════════════════════════════════════════════════

_shutdown = False

def _handle_signal(signum, frame):
    global _shutdown
    log.warning(f"  Signal {signum} received — will checkpoint after this batch ...")
    _shutdown = True

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ═══════════════════════════════════════════════════════════════
#  GPU UTILITIES
# ═══════════════════════════════════════════════════════════════

def gpu_stats() -> str:
    if not torch.cuda.is_available():
        return "CPU mode"
    alloc  = torch.cuda.memory_allocated() / 1e9
    reserv = torch.cuda.memory_reserved()  / 1e9
    total  = torch.cuda.get_device_properties(0).total_memory / 1e9
    free   = total - reserv
    return (f"VRAM  alloc={alloc:.1f}GB  reserved={reserv:.1f}GB  "
            f"free={free:.1f}GB  total={total:.1f}GB")


def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def auto_batch_size(m, tok, target_gb: float = VRAM_LIMIT_GB) -> int:
    """
    Probe VRAM to find largest safe batch for e5-base on L40S.
    e5-base is 2x larger than e5-small, so starts from 4096.
    """
    if not torch.cuda.is_available():
        return 128
    log.info("  Auto-probing batch size for e5-base on L40S 48GB ...")
    for bs in [4096, 2048, 1024, 512, 256, 128, 64]:
        try:
            clear_gpu()
            dummy = [f"passage: benchmark sentence for e5 base probe {i}" for i in range(bs)]
            enc   = tok(
                dummy, padding=True, truncation=True,
                max_length=MAX_LENGTH, return_tensors="pt"
            ).to("cuda")
            with torch.no_grad():
                out = m(**enc)
            used_gb = torch.cuda.memory_allocated() / 1e9
            del enc, out, dummy
            clear_gpu()
            if used_gb <= target_gb:
                log.info(f"  Batch size: {bs:<6}  VRAM used: {used_gb:.1f}GB  OK")
                return bs
        except torch.cuda.OutOfMemoryError:
            clear_gpu()
            log.warning(f"  Batch size: {bs:<6}  OOM — trying smaller")
    log.warning("  Falling back to batch_size=64")
    return 64


# ═══════════════════════════════════════════════════════════════
#  MODEL HELPERS
# ═══════════════════════════════════════════════════════════════

def mean_pooling(model_output, attention_mask):
    """Mean pool token embeddings weighted by attention mask."""
    token_emb = model_output.last_hidden_state          # (B, L, D)
    mask_exp  = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
    return (token_emb * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)


@torch.no_grad()
def encode_texts(texts: list, prefix: str = "passage") -> np.ndarray:
    """
    Encode texts with E5 prefix convention.
    Returns float32 L2-normalised numpy array of shape (N, 768).
    """
    prefixed = [f"{prefix}: {t}" for t in texts]
    encoded  = tokenizer(
        prefixed,
        padding      = True,
        truncation   = True,
        max_length   = MAX_LENGTH,
        return_tensors = "pt"
    ).to(DEVICE)
    output = model(**encoded)
    emb    = mean_pooling(output, encoded["attention_mask"])
    emb    = F.normalize(emb.float(), p=2, dim=1)
    del encoded, output
    return emb.cpu().numpy().astype(np.float32)


# ═══════════════════════════════════════════════════════════════
#  CHECKPOINT HELPERS
# ═══════════════════════════════════════════════════════════════

def save_progress(docids_done: list, n_done: int):
    """Save docids list + progress counter. Fast — no array ops."""
    with open(DOCID_CKPT, "w", encoding="utf-8") as f:
        json.dump(docids_done, f)
    with open(PROGRESS_CKPT, "w", encoding="utf-8") as f:
        json.dump({
            "passages_done" : n_done,
            "saved_at"      : datetime.now().isoformat(),
            "model"         : MODEL_NAME,
            "embed_dim"     : EMBED_DIM,
            "mmap_path"     : str(MMAP_PATH),
        }, f, indent=2)
    log.info(f"  Progress saved — {n_done:,} passages done")


def load_progress():
    """Returns (n_done, docids_done) from checkpoint, or (0, []) if none."""
    if PROGRESS_CKPT.exists() and DOCID_CKPT.exists() and MMAP_PATH.exists():
        with open(PROGRESS_CKPT, encoding="utf-8") as f:
            meta = json.load(f)
        with open(DOCID_CKPT, encoding="utf-8") as f:
            docids_done = json.load(f)
        n_done = meta["passages_done"]

        # Guard: make sure the checkpoint was for the same model/dim
        saved_dim = meta.get("embed_dim", EMBED_DIM)
        if saved_dim != EMBED_DIM:
            log.error(f"  Checkpoint dim={saved_dim} != current EMBED_DIM={EMBED_DIM}. "
                      f"Delete checkpoints and restart.")
            sys.exit(1)

        log.info(f"  Checkpoint found — resuming from {n_done:,} passages")
        log.info(f"  Checkpoint saved at: {meta.get('saved_at', '?')}")
        return n_done, docids_done

    log.info("  No checkpoint — starting fresh")
    return 0, []


def clean_checkpoints():
    for fp in [MMAP_PATH, DOCID_CKPT, PROGRESS_CKPT]:
        if fp.exists():
            fp.unlink()
    log.info("  Checkpoint files cleaned up")


# ═══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

t_pipeline_start = time.time()

log.info("=" * 65)
log.info("  DENSE INDEX BUILDER — multilingual-e5-BASE on L40S 48GB")
log.info("=" * 65)
log.info(f"  Script started : {datetime.now().isoformat()}")
log.info(f"  Log file       : {LOG_FILE}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"  Device         : {DEVICE}")

if DEVICE == "cuda":
    log.info(f"  GPU            : {torch.cuda.get_device_name(0)}")
    log.info(f"  VRAM total     : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    log.info(f"  CUDA version   : {torch.version.cuda}")
    log.info(f"  PyTorch        : {torch.__version__}")
else:
    log.warning("  CUDA not available — running on CPU (will be very slow)")

log.info(f"  Model          : {MODEL_NAME}")
log.info(f"  Embed dim      : {EMBED_DIM}")
log.info(f"  Batch size     : {BATCH_SIZE} (will auto-tune)")
log.info(f"  Max length     : {MAX_LENGTH}")
log.info(f"  Checkpoint     : every {CHECKPOINT_EVERY:,} passages")
log.info(f"  Output dir     : {OUTPUT_DIR}")
log.info(f"  Mmap path      : {MMAP_PATH}")
log.info(f"  mmap size      : ~{8_841_823 * EMBED_DIM * 4 / 1e9:.1f}GB on disk")
log.info("=" * 65)


# ── STEP 1: Load model ───────────────────────────────────────
log.info("\n[ 1 / 6 ]  Loading multilingual-e5-base model ...")
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

if DEVICE == "cuda":
    model = model.half()    # FP16: ~556MB VRAM for e5-base (was ~234MB for e5-small)
    log.info("  FP16 (half precision) enabled")

model.eval()
log.info(f"  Model loaded in {time.time()-t0:.1f}s")
log.info(f"  {gpu_stats()}")

# Auto-tune batch size — e5-base uses ~2x VRAM vs e5-small per token
BATCH_SIZE = auto_batch_size(model, tokenizer, target_gb=VRAM_LIMIT_GB)
log.info(f"  Final batch size: {BATCH_SIZE}")


# ── STEP 2: Load IndicMARCO collection ───────────────────────
log.info("\n[ 2 / 6 ]  Loading IndicMARCO Hindi collection ...")
log.info("  Reading directly from TSV — NOT ir_datasets stream")
t0 = time.time()

if not COLLECTION_TSV.exists():
    log.error(f"  Collection TSV not found: {COLLECTION_TSV}")
    log.error("  Run setup_env_base.sh first to download the dataset.")
    sys.exit(1)

encoding_used = None
for enc in ("utf-8-sig", "utf-8", "latin-1"):
    try:
        df = pd.read_csv(
            COLLECTION_TSV,
            sep          = "\t",
            header       = None,
            names        = ["pid", "text"],
            dtype        = {"pid": str},
            encoding     = enc,
            on_bad_lines = "skip",
            engine       = "python",
        )
        encoding_used = enc
        break
    except (UnicodeDecodeError, UnicodeError):
        continue

if encoding_used is None:
    log.error(f"  Cannot read collection file with any encoding: {COLLECTION_TSV}")
    sys.exit(1)

df.dropna(subset=["text"], inplace=True)
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""].reset_index(drop=True)

if MAX_PASSAGES:
    df = df.iloc[:MAX_PASSAGES].reset_index(drop=True)
    log.info(f"  Limited to MAX_PASSAGES={MAX_PASSAGES:,}")

log.info(f"  Encoding       : {encoding_used}")
log.info(f"  Total passages : {len(df):,}")
log.info(f"  Load time      : {time.time()-t0:.1f}s")

all_docids   = df["pid"].tolist()
all_passages = df["text"].tolist()
N_PASSAGES   = len(all_passages)

del df
gc.collect()
log.info("  DataFrame freed — RAM released")


# ── STEP 3: Encode passages → mmap ───────────────────────────
log.info(f"\n[ 3 / 6 ]  Encoding {N_PASSAGES:,} passages ...")
log.info(f"  mmap file: {MMAP_PATH}")
log.info(f"  mmap size on disk: ~{N_PASSAGES * EMBED_DIM * 4 / 1e9:.1f}GB")

start_idx, encoded_docids = load_progress()
log.info(f"  Passages remaining: {N_PASSAGES - start_idx:,}")

# Create or reopen mmap — NEVER loads full array into RAM
if MMAP_PATH.exists() and start_idx > 0:
    log.info(f"  Reopening mmap for resume at offset {start_idx:,}")
    emb_mmap = np.memmap(
        str(MMAP_PATH), dtype="float32", mode="r+",
        shape=(N_PASSAGES, EMBED_DIM)
    )
else:
    log.info(f"  Creating new mmap: shape=({N_PASSAGES}, {EMBED_DIM})")
    emb_mmap = np.memmap(
        str(MMAP_PATH), dtype="float32", mode="w+",
        shape=(N_PASSAGES, EMBED_DIM)
    )

t_encode_start = time.time()
oom_count      = 0
current_bs     = BATCH_SIZE
i              = 0

remaining_passages = all_passages[start_idx:]
remaining_docids   = all_docids[start_idx:]
n_remaining        = len(remaining_passages)

pbar = tqdm(
    total         = n_remaining,
    desc          = "  Encoding",
    unit          = "passage",
    file          = sys.stdout,
    dynamic_ncols = True,
)

while i < n_remaining:

    # ── Graceful shutdown checkpoint ─────────────────────────
    if _shutdown:
        log.warning("  Shutdown signal — saving checkpoint before exit ...")
        emb_mmap.flush()
        save_progress(encoded_docids, start_idx + i)
        log.info("  Safe checkpoint saved. Exiting.")
        sys.exit(0)

    batch_texts  = remaining_passages[i : i + current_bs]
    batch_docids = remaining_docids[i   : i + current_bs]
    actual_bs    = len(batch_texts)

    # ── OOM-safe encode with progressive batch halving ───────
    success = False
    try_bs  = current_bs

    while not success and try_bs >= 32:
        try:
            clear_gpu()
            if try_bs >= actual_bs:
                batch_emb = encode_texts(batch_texts, prefix="passage")
            else:
                # Split current batch into try_bs sub-chunks
                parts = []
                for j in range(0, actual_bs, try_bs):
                    clear_gpu()
                    parts.append(
                        encode_texts(batch_texts[j : j + try_bs], prefix="passage")
                    )
                batch_emb = np.vstack(parts).astype(np.float32)
            success = True

        except torch.cuda.OutOfMemoryError:
            oom_count += 1
            clear_gpu()
            try_bs = try_bs // 2
            log.warning(f"  OOM — batch reduced to {try_bs}")

    if not success:
        log.error("  Cannot encode even at batch=32. Saving checkpoint and exiting.")
        emb_mmap.flush()
        save_progress(encoded_docids, start_idx + i)
        sys.exit(1)

    # ── Write to mmap — ZERO RAM accumulation ────────────────
    write_start = start_idx + i
    write_end   = write_start + len(batch_emb)
    emb_mmap[write_start : write_end] = batch_emb
    encoded_docids.extend(batch_docids[:len(batch_emb)])

    pbar.update(actual_bs)
    i += actual_bs

    # ── Periodic checkpoint ───────────────────────────────────
    total_done = start_idx + i
    if total_done % CHECKPOINT_EVERY < current_bs:
        emb_mmap.flush()
        save_progress(encoded_docids, total_done)
        elapsed = time.time() - t_encode_start
        speed   = i / elapsed if elapsed > 0 else 1
        eta_min = (n_remaining - i) / speed / 60
        log.info(f"  Speed: {speed:.0f} pass/sec | ETA: {eta_min:.1f} min | {gpu_stats()}")
        clear_gpu()

pbar.close()

# Final flush
emb_mmap.flush()
save_progress(encoded_docids, N_PASSAGES)
encode_time = time.time() - t_encode_start
log.info(f"\n  Encoding complete in {encode_time / 60:.1f} min")
log.info(f"  Throughput     : {N_PASSAGES / encode_time:.0f} passages/sec")
log.info(f"  OOM events     : {oom_count}")
log.info(f"  {gpu_stats()}")

del all_passages, all_docids, remaining_passages, remaining_docids
gc.collect()
log.info("  Passage lists freed from RAM")


# ── STEP 4: Build FAISS index — stream from mmap ─────────────
log.info("\n[ 4 / 6 ]  Building FAISS index ...")
log.info("  Streaming 500K chunks from mmap — RAM stays flat")
log.info(f"  Total vectors to add: {N_PASSAGES:,}  dim={EMBED_DIM}")
t0 = time.time()

index = faiss.IndexFlatIP(EMBED_DIM)   # inner product on L2-normed = cosine sim
CHUNK = 500_000

for chunk_start in tqdm(
    range(0, N_PASSAGES, CHUNK),
    desc="  Building FAISS",
    file=sys.stdout
):
    chunk_end  = min(chunk_start + CHUNK, N_PASSAGES)
    chunk_data = np.ascontiguousarray(
        emb_mmap[chunk_start:chunk_end].astype(np.float32)
    )
    index.add(chunk_data)
    del chunk_data
    gc.collect()

log.info(f"  Index built in {time.time()-t0:.1f}s")
log.info(f"  Vectors    : {index.ntotal:,}")
log.info(f"  Dimension  : {EMBED_DIM}")

# Sanity check
assert index.ntotal == len(encoded_docids), (
    f"MISMATCH: {index.ntotal} index vectors vs {len(encoded_docids)} docids"
)
log.info("  Docid count matches index size")

# Save index and docid map
faiss.write_index(index, str(INDEX_PATH))
with open(DOCID_MAP_PATH, "w", encoding="utf-8") as f:
    json.dump(encoded_docids, f)

log.info(f"  Index saved     → {INDEX_PATH}  ({INDEX_PATH.stat().st_size/1e6:.0f} MB)")
log.info(f"  Docid map saved → {DOCID_MAP_PATH}  ({DOCID_MAP_PATH.stat().st_size/1e6:.0f} MB)")

# Free mmap and checkpoint files
del emb_mmap
clean_checkpoints()
clear_gpu()


# ── STEP 5: Load queries + qrels → retrieve ──────────────────
log.info(f"\n[ 5 / 6 ]  Retrieval top-{TOP_K} ...")

# Load queries
queries = {}
if not QUERIES_FILE.exists():
    log.error(f"  Queries file not found: {QUERIES_FILE}")
    log.error("  Run setup_env_base.sh to export queries from ir_datasets.")
    sys.exit(1)

with open(QUERIES_FILE, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split("\t", 1)
            if len(parts) == 2:
                queries[parts[0].strip()] = parts[1].strip()
log.info(f"  Queries loaded : {len(queries):,}")

# Load qrels
qrels = defaultdict(dict)
if not QRELS_FILE.exists():
    log.error(f"  Qrels file not found: {QRELS_FILE}")
    sys.exit(1)

with open(QRELS_FILE, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 4:
            qid, _, pid, rel = parts
            qrels[qid][pid] = int(rel)
log.info(f"  Qrels loaded   : {len(qrels):,} queries")

# Retrieve top-K for every query
t0  = time.time()
run = {}

for qid, query_text in tqdm(queries.items(), desc="  Retrieving", file=sys.stdout):
    q_emb           = encode_texts([query_text], prefix="query")
    scores, indices = index.search(np.array(q_emb, dtype=np.float32), TOP_K)
    run[qid] = {}
    for idx, score in zip(indices[0], scores[0]):
        if 0 <= int(idx) < len(encoded_docids):
            run[qid][encoded_docids[int(idx)]] = float(score)

log.info(f"  Retrieval done in {time.time()-t0:.1f}s")

# Save TREC run file
with open(RUN_FILE, "w", encoding="utf-8") as f:
    for qid in run:
        ranked = sorted(run[qid].items(), key=lambda x: x[1], reverse=True)
        for rank, (pid, score) in enumerate(ranked, 1):
            f.write(f"{qid}\tQ0\t{pid}\t{rank}\t{score:.6f}\te5-base\n")
log.info(f"  Run file saved → {RUN_FILE}")


# ── STEP 6: Evaluate MRR, Recall, NDCG ───────────────────────
log.info(f"\n[ 6 / 6 ]  Evaluating MRR@{EVAL_K}, Recall@{EVAL_K}, NDCG@{EVAL_K} ...")

def dcg(rel_list: list) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rel_list))


def evaluate(qrels_dict, run_dict, k: int = 10) -> dict:
    mrr = recall = ndcg = 0.0
    n   = 0

    for qid, relevant in tqdm(qrels_dict.items(), desc="  Evaluating", file=sys.stdout):
        if qid not in run_dict:
            continue
        rel_set = {pid for pid, r in relevant.items() if r >= 1}
        if not rel_set:
            continue

        ranked_k = [
            pid for pid, _ in
            sorted(run_dict[qid].items(), key=lambda x: x[1], reverse=True)
        ][:k]

        # MRR
        for rank, pid in enumerate(ranked_k, 1):
            if pid in rel_set:
                mrr += 1.0 / rank
                break

        # Recall
        recall += len(set(ranked_k) & rel_set) / len(rel_set)

        # NDCG
        gains = [relevant.get(pid, 0) for pid in ranked_k]
        ideal = sorted(relevant.values(), reverse=True)[:k]
        ndcg += dcg(gains) / (dcg(ideal) + 1e-9)
        n    += 1

    if n == 0:
        log.warning("  No overlapping queries between run and qrels!")
        log.warning("  Check that query IDs in queries file match qrels file.")
        return {}

    results = {
        "model"        : MODEL_NAME,
        "embed_dim"    : EMBED_DIM,
        "gpu"          : torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
        "batch_size"   : BATCH_SIZE,
        "max_length"   : MAX_LENGTH,
        "num_queries"  : n,
        f"MRR@{k}"     : round(mrr    / n, 4),
        f"Recall@{k}"  : round(recall / n, 4),
        f"NDCG@{k}"    : round(ndcg   / n, 4),
    }

    log.info("\n" + "=" * 65)
    log.info("  EVALUATION RESULTS — multilingual-e5-base on L40S")
    log.info("=" * 65)
    log.info(f"  Model        : {MODEL_NAME}")
    log.info(f"  Embed dim    : {EMBED_DIM}")
    log.info(f"  GPU          : {results['gpu']}")
    log.info(f"  Queries      : {n}")
    log.info(f"  MRR@{k}      : {results[f'MRR@{k}']:.4f}")
    log.info(f"  Recall@{k}   : {results[f'Recall@{k}']:.4f}")
    log.info(f"  NDCG@{k}     : {results[f'NDCG@{k}']:.4f}")
    log.info("=" * 65)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info(f"  Results saved → {RESULTS_FILE}")
    return results


metrics = evaluate(qrels, run, k=EVAL_K)


# ── Final Summary ─────────────────────────────────────────────
total_time = time.time() - t_pipeline_start

log.info("\n" + "=" * 65)
log.info("  OUTPUT FILES")
log.info("=" * 65)
for fp in sorted(OUTPUT_DIR.iterdir()):
    if fp.is_file():
        mb = fp.stat().st_size / (1024 ** 2)
        log.info(f"  {fp.name:<48} {mb:7.1f} MB")
log.info("=" * 65)
log.info(f"\n  PIPELINE COMPLETE in {total_time / 60:.1f} min")
log.info(f"  Log saved → {LOG_FILE}")
log.info("\n  To load index later:")
log.info("    import faiss, json")
log.info(f"    index  = faiss.read_index('{INDEX_PATH}')")
log.info(f"    docids = json.load(open('{DOCID_MAP_PATH}'))")
log.info(f"    # query embedding must be dim={EMBED_DIM} (e5-base)")
log.info("    scores, idx = index.search(query_emb, top_k=100)")
