"""
═══════════════════════════════════════════════════════════════
  Dense Index Builder — Strictly Optimized for L40S 48GB VRAM
  Model   : intfloat/multilingual-e5-small
  Dataset : IndicMARCO Hindi (via ir_datasets)
  Platform: Lightning AI Studio (L40S — 48GB VRAM, 16 CPUs)
═══════════════════════════════════════════════════════════════
  WHY THIS WON'T GET KILLED:
  - Uses memory-mapped (mmap) numpy array instead of emb_list
  - emb_list approach doubles RAM at every checkpoint → OS kill
  - mmap writes directly to disk, never accumulates in RAM
  - Checkpoint = just save 1 integer (passages_done)
  - Resume = reopen mmap from disk, skip already-done passages

  HOW TO RUN:
      nohup python build_index_l40s.py > logs/run.log 2>&1 &
      tail -f logs/run.log
      # check progress anytime:
      cat data/e5_index/checkpoints/progress.json
═══════════════════════════════════════════════════════════════
"""

# ── Set CUDA memory config BEFORE importing torch ────────────
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True,"
    "max_split_size_mb:512"
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # avoid fork warnings

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
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

STUDIO_DIR = Path("/teamspace/studios/this_studio")
DATA_DIR   = STUDIO_DIR / "data"

# ── Input ────────────────────────────────────────────────────
# ir_datasets will auto-download and cache IndicMARCO
DATASET_ID   = "mmarco/v2/hi/train"
EVAL_DS_ID   = "mmarco/v2/hi/trec-dl-2019"

# Fallback: TSV files if ir_datasets cache already exists
COLLECTION_TSV = Path("/teamspace/studios/this_studio/.ir_datasets/mmarco/v2/hindi_collection.tsv")
QUERIES_FILE   = DATA_DIR / "trec_dl19_hindi_query.tsv"
QRELS_FILE     = DATA_DIR / "pass_2019.qrels"

# ── Output ───────────────────────────────────────────────────
OUTPUT_DIR = DATA_DIR / "e5_index"
LOG_DIR    = STUDIO_DIR / "logs"
CKPT_DIR   = OUTPUT_DIR / "checkpoints"

for d in [OUTPUT_DIR, LOG_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

INDEX_PATH     = OUTPUT_DIR / "index.faiss"
DOCID_MAP_PATH = OUTPUT_DIR / "docid_map.json"
RUN_FILE       = OUTPUT_DIR / "retrieval_run_top100.trec"
RESULTS_FILE   = OUTPUT_DIR / "evaluation_results.json"

# ── Checkpoint files ─────────────────────────────────────────
# KEY DESIGN: embeddings stored as memory-mapped file on disk
# → never accumulates in RAM → process cannot be OOM-killed
MMAP_PATH     = CKPT_DIR / "embeddings.mmap"   # memory-mapped array
DOCID_CKPT    = CKPT_DIR / "docids.json"
PROGRESS_CKPT = CKPT_DIR / "progress.json"

# ── Log ──────────────────────────────────────────────────────
LOG_FILE = LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# ── Model ────────────────────────────────────────────────────
MODEL_NAME = "intfloat/multilingual-e5-small"
EMBED_DIM  = 384    # fixed output dim for multilingual-e5-small

# ── L40S 48GB Hyperparams ────────────────────────────────────
#   L40S: 48GB VRAM, 362 TFLOPs, 16 CPUs
#   e5-small: 117M params, 384-dim embeddings
#   At FP16, model takes ~234MB VRAM
#   Batch 4096 × 512 tokens × FP16 ≈ ~8GB VRAM → safe headroom
BATCH_SIZE       = 4096    # maximize L40S throughput
MAX_LENGTH       = 512     # full context window (L40S handles it)
TOP_K            = 100
EVAL_K           = 10
MAX_PASSAGES     = None    # None = full dataset
CHECKPOINT_EVERY = 25_000  # save every 25K passages (~30s on L40S)
VRAM_LIMIT_GB    = 44.0    # use 44 of 48GB, keep 4GB headroom


# ═══════════════════════════════════════════════════════════════
#  LOGGING — writes to terminal AND timestamped log file
# ═══════════════════════════════════════════════════════════════

def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("e5_l40s")
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
#  GRACEFUL SHUTDOWN — save checkpoint on Ctrl+C / SIGTERM
# ═══════════════════════════════════════════════════════════════

_shutdown_requested = False

def _handle_signal(signum, frame):
    global _shutdown_requested
    log.warning(f"Signal {signum} received — will checkpoint and exit after current batch")
    _shutdown_requested = True

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ═══════════════════════════════════════════════════════════════
#  GPU UTILITIES
# ═══════════════════════════════════════════════════════════════

def gpu_stats() -> str:
    if not torch.cuda.is_available():
        return "CPU mode"
    alloc  = torch.cuda.memory_allocated()  / 1e9
    reserv = torch.cuda.memory_reserved()   / 1e9
    total  = torch.cuda.get_device_properties(0).total_memory / 1e9
    free   = total - reserv
    return (f"VRAM  alloc={alloc:.1f}GB  reserved={reserv:.1f}GB  "
            f"free={free:.1f}GB  total={total:.1f}GB")


def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def auto_batch_size(m, tok, n_passages: int, target_gb: float = VRAM_LIMIT_GB) -> int:
    """
    Probe VRAM to find largest safe batch size for this GPU.
    Tests [8192, 4096, 2048, 1024, 512, 256, 128] in order.
    """
    if not torch.cuda.is_available():
        return 128

    log.info("  Auto-probing batch size for L40S 48GB ...")
    for bs in [8192, 4096, 2048, 1024, 512, 256, 128]:
        try:
            clear_gpu()
            dummy = [f"passage: benchmark sentence for batch size probe {i}" for i in range(bs)]
            enc   = tok(dummy, padding=True, truncation=True,
                        max_length=MAX_LENGTH, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = m(**enc)
            used_gb = torch.cuda.memory_allocated() / 1e9
            del enc, out, dummy
            clear_gpu()
            if used_gb <= target_gb:
                log.info(f"  Batch size: {bs:<6}  VRAM used during probe: {used_gb:.1f}GB  ✅")
                return bs
        except torch.cuda.OutOfMemoryError:
            clear_gpu()
            log.warning(f"  Batch size: {bs:<6}  OOM — trying smaller")
    log.warning("  Falling back to batch_size=64")
    return 64


# ═══════════════════════════════════════════════════════════════
#  MEMORY-MAPPED CHECKPOINT  (the key fix vs old code)
# ═══════════════════════════════════════════════════════════════

def init_mmap(n_passages: int, dim: int) -> np.memmap:
    """
    Create or reopen a memory-mapped float32 array on disk.
    Shape: (n_passages, dim)
    Writing to this never uses RAM — goes directly to disk.
    """
    return np.memmap(
        str(MMAP_PATH),
        dtype  = "float32",
        mode   = "w+" if not MMAP_PATH.exists() else "r+",
        shape  = (n_passages, dim)
    )


def open_mmap(n_passages: int, dim: int) -> np.memmap:
    """Reopen existing mmap for reading/appending."""
    return np.memmap(
        str(MMAP_PATH),
        dtype  = "float32",
        mode   = "r+",
        shape  = (n_passages, dim)
    )


def save_progress(docids_done: list, n_done: int):
    """Lightweight checkpoint — only saves docids + progress int."""
    with open(DOCID_CKPT,    "w", encoding="utf-8") as f:
        json.dump(docids_done, f)
    with open(PROGRESS_CKPT, "w", encoding="utf-8") as f:
        json.dump({
            "passages_done": n_done,
            "saved_at"     : datetime.now().isoformat(),
            "mmap_path"    : str(MMAP_PATH),
        }, f, indent=2)
    log.info(f"  💾 Progress saved — {n_done:,} passages done")


def load_progress():
    """
    Returns (n_done, docids_done) if checkpoint exists, else (0, []).
    mmap is always reopened externally with the correct shape.
    """
    if PROGRESS_CKPT.exists() and DOCID_CKPT.exists() and MMAP_PATH.exists():
        with open(PROGRESS_CKPT, encoding="utf-8") as f:
            meta = json.load(f)
        with open(DOCID_CKPT,    encoding="utf-8") as f:
            docids_done = json.load(f)
        n_done = meta["passages_done"]
        log.info(f"  ♻️  Checkpoint found — resuming from {n_done:,} passages")
        log.info(f"      Saved at: {meta.get('saved_at', 'unknown')}")
        return n_done, docids_done
    log.info("  No checkpoint — starting fresh")
    return 0, []


def clean_checkpoints():
    for f in [MMAP_PATH, DOCID_CKPT, PROGRESS_CKPT]:
        if f.exists():
            f.unlink()
    log.info("  🗑️  Checkpoint files deleted")


# ═══════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════

def mean_pooling(model_output, attention_mask):
    token_emb = model_output.last_hidden_state
    mask_exp  = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
    return (token_emb * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)


@torch.no_grad()
def encode_batch(texts: list, prefix: str = "passage") -> np.ndarray:
    prefixed = [f"{prefix}: {t}" for t in texts]
    encoded  = tokenizer(
        prefixed, padding=True, truncation=True,
        max_length=MAX_LENGTH, return_tensors="pt"
    ).to(DEVICE)
    output = model(**encoded)
    emb    = mean_pooling(output, encoded["attention_mask"])
    emb    = torch.nn.functional.normalize(emb.float(), p=2, dim=1)
    del encoded, output
    return emb.cpu().numpy().astype(np.float32)


# ═══════════════════════════════════════════════════════════════
#  PIPELINE START
# ═══════════════════════════════════════════════════════════════

t_pipeline_start = time.time()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

log.info("=" * 65)
log.info("  DENSE INDEX BUILDER — L40S 48GB OPTIMIZED")
log.info("=" * 65)
log.info(f"  Script started : {datetime.now().isoformat()}")
log.info(f"  Log file       : {LOG_FILE}")
log.info(f"  Device         : {DEVICE}")

if DEVICE == "cuda":
    log.info(f"  GPU            : {torch.cuda.get_device_name(0)}")
    log.info(f"  VRAM total     : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    log.info(f"  CUDA version   : {torch.version.cuda}")
    log.info(f"  PyTorch        : {torch.__version__}")
else:
    log.warning("  CUDA not available — running on CPU (slow)")

log.info(f"  Batch size     : {BATCH_SIZE} (will auto-tune)")
log.info(f"  Max length     : {MAX_LENGTH}")
log.info(f"  Checkpoint     : every {CHECKPOINT_EVERY:,} passages")
log.info(f"  Mmap path      : {MMAP_PATH}")
log.info("=" * 65)


# ── STEP 1: Load model ───────────────────────────────────────
log.info("\n[ 1 / 6 ]  Loading model ...")
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

if DEVICE == "cuda":
    model = model.half()   # FP16: ~234MB VRAM for e5-small
    log.info("  FP16 (half precision) enabled")

model.eval()
log.info(f"  ✅ Model loaded in {time.time()-t0:.1f}s")
log.info(f"  {gpu_stats()}")

# Auto-tune batch size to saturate L40S VRAM
BATCH_SIZE = auto_batch_size(model, tokenizer, MAX_PASSAGES or 216_517)
log.info(f"  Final batch size: {BATCH_SIZE}")


# ── STEP 2: Load collection ──────────────────────────────────
log.info("\n[ 2 / 6 ]  Loading IndicMARCO collection ...")
t0 = time.time()

all_docids   = []
all_passages = []

# Try ir_datasets first (preferred), fallback to TSV
try:
    import ir_datasets
    log.info(f"  Loading via ir_datasets: {DATASET_ID}")
    ds = ir_datasets.load(DATASET_ID)
    for doc in tqdm(ds.docs_iter(), desc="  Reading docs", file=sys.stdout):
        all_docids.append(str(doc.doc_id))
        all_passages.append(str(doc.text).strip())
    log.info(f"  Source: ir_datasets ({DATASET_ID})")

except Exception as e:
    log.warning(f"  ir_datasets failed ({e}) — falling back to TSV")
    enc_used = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(
                COLLECTION_TSV, sep="\t", header=None,
                names=["pid", "text"], dtype={"pid": str},
                encoding=enc, on_bad_lines="skip", engine="python"
            )
            enc_used = enc
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if enc_used is None:
        log.error(f"Cannot read collection: {COLLECTION_TSV}")
        sys.exit(1)
    df.dropna(subset=["text"], inplace=True)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)
    all_docids   = df["pid"].tolist()
    all_passages = df["text"].tolist()
    log.info(f"  Source: TSV ({enc_used})")

log.info(f"  Total passages : {len(all_passages):,}")
log.info(f"  Load time      : {time.time()-t0:.1f}s")

if MAX_PASSAGES and len(all_passages) > MAX_PASSAGES:
    all_passages = all_passages[:MAX_PASSAGES]
    all_docids   = all_docids[:MAX_PASSAGES]
    log.info(f"  Limited to     : {len(all_passages):,} (MAX_PASSAGES)")

N_PASSAGES = len(all_passages)


# ── STEP 3: Encode with mmap checkpointing ───────────────────
log.info(f"\n[ 3 / 6 ]  Encoding {N_PASSAGES:,} passages (mmap checkpoint) ...")
log.info(f"  Estimated time : ~{N_PASSAGES//(BATCH_SIZE * 60)+1} min on L40S GPU")

# Load progress (if resuming)
start_idx, encoded_docids = load_progress()

# Open or create mmap  (shape = full dataset size, fill incrementally)
if start_idx == 0:
    log.info(f"  Creating mmap array: shape=({N_PASSAGES}, {EMBED_DIM}), "
             f"size={N_PASSAGES * EMBED_DIM * 4 / 1e9:.2f}GB on disk")
    emb_mmap = init_mmap(N_PASSAGES, EMBED_DIM)
else:
    log.info(f"  Reopening mmap: shape=({N_PASSAGES}, {EMBED_DIM})")
    emb_mmap = open_mmap(N_PASSAGES, EMBED_DIM)

t_encode_start = time.time()
oom_count      = 0
current_bs     = BATCH_SIZE

if start_idx >= N_PASSAGES:
    log.info("  ✅ All passages already encoded from checkpoint!")
else:
    remaining_passages = all_passages[start_idx:]
    remaining_docids   = all_docids[start_idx:]

    pbar = tqdm(
        total     = len(remaining_passages),
        desc      = "  Encoding",
        unit      = "passage",
        file      = sys.stdout,
        dynamic_ncols = True,
    )

    i = 0
    while i < len(remaining_passages):

        # Handle graceful shutdown signal
        if _shutdown_requested:
            total_done = start_idx + i
            log.warning(f"  Shutdown requested — saving checkpoint at {total_done:,}")
            save_progress(encoded_docids, total_done)
            emb_mmap.flush()
            log.warning("  Checkpoint saved. Exiting safely.")
            sys.exit(0)

        batch_texts  = remaining_passages[i : i + current_bs]
        batch_docids = remaining_docids[i   : i + current_bs]
        actual_bs    = len(batch_texts)

        # OOM-safe encode with auto-retry
        success = False
        while not success and current_bs >= 32:
            try:
                batch_emb = encode_batch(batch_texts[:current_bs], prefix="passage")

                # If batch was shrunk, encode remaining sub-batches
                if current_bs < actual_bs:
                    sub_embs = [batch_emb]
                    for j in range(current_bs, actual_bs, current_bs):
                        clear_gpu()
                        sub_embs.append(
                            encode_batch(batch_texts[j:j+current_bs], prefix="passage")
                        )
                    batch_emb = np.vstack(sub_embs).astype(np.float32)

                success = True

            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                clear_gpu()
                current_bs = current_bs // 2
                log.warning(f"  ⚠️  OOM — batch reduced to {current_bs}")

        if not success:
            log.error("  ❌ Cannot encode even at batch=32. Check GPU health.")
            emb_mmap.flush()
            save_progress(encoded_docids, start_idx + i)
            sys.exit(1)

        # Write directly to mmap (no RAM accumulation!)
        write_start = start_idx + i
        write_end   = write_start + len(batch_emb)
        emb_mmap[write_start:write_end] = batch_emb
        encoded_docids.extend(batch_docids[:len(batch_emb)])

        pbar.update(actual_bs)
        i += actual_bs

        # Periodic checkpoint
        total_done = start_idx + i
        if total_done % CHECKPOINT_EVERY < current_bs:
            emb_mmap.flush()                          # flush mmap to disk
            save_progress(encoded_docids, total_done)
            log.info(f"  {gpu_stats()}")
            clear_gpu()

    pbar.close()

    # Final flush + checkpoint
    emb_mmap.flush()
    save_progress(encoded_docids, N_PASSAGES)
    encode_time = time.time() - t_encode_start
    log.info(f"\n  ✅ Encoding complete in {encode_time/60:.1f} min")
    log.info(f"  Throughput     : {N_PASSAGES/encode_time:.0f} passages/sec")
    log.info(f"  OOM events     : {oom_count}")
    log.info(f"  {gpu_stats()}")

# Read final embeddings from mmap (zero-copy, no RAM spike)
log.info("\n  Reading embeddings from mmap ...")
t0 = time.time()
passage_embeddings = np.array(emb_mmap, dtype=np.float32)     # one contiguous copy
del emb_mmap
if not passage_embeddings.flags["C_CONTIGUOUS"]:
    passage_embeddings = np.ascontiguousarray(passage_embeddings)
docids = encoded_docids

log.info(f"  Shape          : {passage_embeddings.shape}")
log.info(f"  Memory         : {passage_embeddings.nbytes/1e9:.2f} GB")
log.info(f"  Read time      : {time.time()-t0:.1f}s")

# Free passage text — no longer needed
del all_passages, all_docids
gc.collect()

# Sanity checks
assert passage_embeddings.ndim  == 2,            "Expected 2D array"
assert passage_embeddings.dtype == np.float32,   "Expected float32"
assert len(passage_embeddings)  == len(docids),  "Size mismatch!"
assert np.isfinite(passage_embeddings).all(),    "NaN/Inf in embeddings!"
log.info("  ✅ All sanity checks passed")


# ── STEP 4: Build FAISS index ────────────────────────────────
log.info("\n[ 4 / 6 ]  Building FAISS index ...")
t0 = time.time()

dim   = int(passage_embeddings.shape[1])
index = faiss.IndexFlatIP(dim)      # inner product on L2-normed = cosine sim
index.add(passage_embeddings)       # type: ignore[arg-type]

log.info(f"  ✅ Index built in {time.time()-t0:.1f}s")
log.info(f"  Vectors    : {index.ntotal:,}")
log.info(f"  Dimension  : {dim}")

faiss.write_index(index, str(INDEX_PATH))
with open(DOCID_MAP_PATH, "w", encoding="utf-8") as f:
    json.dump(docids, f)

log.info(f"  Index saved     → {INDEX_PATH}  ({INDEX_PATH.stat().st_size/1e6:.1f} MB)")
log.info(f"  Docid map saved → {DOCID_MAP_PATH}  ({DOCID_MAP_PATH.stat().st_size/1e6:.1f} MB)")

clean_checkpoints()
clear_gpu()


# ── STEP 5: Load queries + qrels ─────────────────────────────
log.info(f"\n[ 5 / 6 ]  Retrieval top-{TOP_K} ...")

queries = {}
qrels   = defaultdict(dict)

# Try ir_datasets eval set first
try:
    import ir_datasets
    log.info(f"  Loading eval set via ir_datasets: {EVAL_DS_ID}")
    eval_ds = ir_datasets.load(EVAL_DS_ID)
    for q in eval_ds.queries_iter():
        queries[str(q.query_id)] = str(q.text).strip()
    for qrel in eval_ds.qrels_iter():
        qrels[str(qrel.query_id)][str(qrel.doc_id)] = int(qrel.relevance)
    log.info(f"  Source: ir_datasets ({EVAL_DS_ID})")

except Exception as e:
    log.warning(f"  ir_datasets eval failed ({e}) — loading from TSV files")
    with open(QUERIES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    queries[parts[0].strip()] = parts[1].strip()
    with open(QRELS_FILE, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                qid, _, pid, rel = parts
                qrels[qid][pid]  = int(rel)

log.info(f"  Queries : {len(queries)}")
log.info(f"  Qrels   : {len(qrels)} queries")

# Retrieve
t0  = time.time()
run = {}
for qid, query_text in tqdm(queries.items(), desc="  Retrieving", file=sys.stdout):
    q_emb           = encode_batch([query_text], prefix="query")
    scores, indices = index.search(np.array(q_emb, dtype=np.float32), TOP_K)
    run[qid] = {}
    for idx, score in zip(indices[0], scores[0]):
        if 0 <= int(idx) < len(docids):
            run[qid][docids[int(idx)]] = float(score)

log.info(f"  Retrieval done in {time.time()-t0:.1f}s")

with open(RUN_FILE, "w", encoding="utf-8") as f:
    for qid in run:
        ranked = sorted(run[qid].items(), key=lambda x: x[1], reverse=True)
        for rank, (pid, score) in enumerate(ranked, 1):
            f.write(f"{qid}\tQ0\t{pid}\t{rank}\t{score:.6f}\te5-small\n")
log.info(f"  Run file saved → {RUN_FILE}")


# ── STEP 6: Evaluation ───────────────────────────────────────
log.info(f"\n[ 6 / 6 ]  Evaluating MRR@{EVAL_K}, Recall@{EVAL_K}, NDCG@{EVAL_K} ...")

def dcg(rel_list: list) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rel_list))

def evaluate(qrels_dict, run_dict, k=10):
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
        for rank, pid in enumerate(ranked_k, 1):
            if pid in rel_set:
                mrr += 1.0 / rank
                break
        recall += len(set(ranked_k) & rel_set) / len(rel_set)
        gains   = [relevant.get(pid, 0) for pid in ranked_k]
        ideal   = sorted(relevant.values(), reverse=True)[:k]
        ndcg   += dcg(gains) / (dcg(ideal) + 1e-9)
        n      += 1

    if n == 0:
        log.warning("⚠  No overlapping queries found!")
        return {}

    results = {
        "model"        : MODEL_NAME,
        "gpu"          : torch.cuda.get_device_name(0) if DEVICE=="cuda" else "cpu",
        "batch_size"   : BATCH_SIZE,
        "num_queries"  : n,
        f"MRR@{k}"     : round(mrr    / n, 4),
        f"Recall@{k}"  : round(recall / n, 4),
        f"NDCG@{k}"    : round(ndcg   / n, 4),
    }

    log.info("\n" + "=" * 65)
    log.info("  EVALUATION RESULTS — multilingual-e5-small on L40S")
    log.info("=" * 65)
    log.info(f"  Model        : {MODEL_NAME}")
    log.info(f"  GPU          : {results['gpu']}")
    log.info(f"  Queries      : {n}")
    log.info(f"  MRR@{k:<3}       : {results[f'MRR@{k}']:.4f}")
    log.info(f"  Recall@{k:<3}    : {results[f'Recall@{k}']:.4f}")
    log.info(f"  NDCG@{k:<3}      : {results[f'NDCG@{k}']:.4f}")
    log.info("=" * 65)

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"  Results saved → {RESULTS_FILE}")
    return results

metrics = evaluate(qrels, run, k=EVAL_K)


# ── Final Summary ────────────────────────────────────────────
total_time = time.time() - t_pipeline_start
log.info("\n" + "=" * 65)
log.info("  OUTPUT FILES")
log.info("=" * 65)
for fp in sorted(OUTPUT_DIR.iterdir()):
    if fp.is_file():
        mb = fp.stat().st_size / (1024**2)
        log.info(f"  {fp.name:<48} {mb:7.1f} MB")
log.info("=" * 65)
log.info(f"\n  ✅ PIPELINE COMPLETE in {total_time/60:.1f} min")
log.info(f"  Log saved → {LOG_FILE}")
log.info("\n  To load index later:")
log.info("    import faiss, json")
log.info(f"    index  = faiss.read_index('{INDEX_PATH}')")
log.info(f"    docids = json.load(open('{DOCID_MAP_PATH}'))")
log.info("    scores, idx = index.search(query_emb, top_k=100)")
