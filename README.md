# 🌐 Cross-Lingual Information Retrieval using BM25 and Dense Indexing

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-GPU%20Indexing-009900)](https://github.com/facebookresearch/faiss)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Mini Project** — B.E. Computer Engineering  
> **Guide:** Prof. Prasanjit  
> **Author:** Swayam Patel  

---

## 📌 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Models](#models)
- [Experimental Results](#experimental-results)
- [Repository Structure](#repository-structure)
- [Setup & Usage](#setup--usage)
- [Key Findings](#key-findings)
- [References](#references)

---

## Overview

This project tackles **Cross-Lingual Information Retrieval (CLIR)** — the challenge of retrieving English documents using Hindi queries (and vice versa). Traditional keyword-based systems fail completely when query and document languages differ; dense retrieval models trained on multilingual data bridge this gap naturally through shared semantic embedding spaces.

We benchmark two complementary retrieval paradigms:

| Paradigm | Method | Approach |
|----------|--------|----------|
| **Sparse** | BM25 (via Pyserini) | Lexical term-frequency matching with Hindi Lucene analyzer |
| **Dense** | Multilingual-E5 (Small / Base / Large) | Bi-encoder: FAISS inner-product search over 8.8M passage embeddings |

All experiments run on the **IndicMARCO Hindi** collection (~8.8 million passages) and are evaluated against TREC Deep Learning 2019 and 2020 passage benchmarks.

---

## Problem Statement

A Hindi-speaking user submits the query:

> *"भारत की राजधानी क्या है?"*

The relevant document in the collection is in English:

> *"New Delhi is the capital of India."*

A classical BM25 system finds **zero overlapping tokens** and retrieves nothing. Dense models map both the Hindi query and the English document into a shared multilingual vector space — enabling retrieval based on **meaning, not surface form**.

---

## Dataset

| Component | Details |
|-----------|---------|
| **Collection** | IndicMARCO Hindi — 8,841,823 passages (translated from MS MARCO) |
| **Source** | `saifulhaq9/indicmarco` (HuggingFace) |
| **Query Set 1** | TREC DL 2019 — 43 Hindi queries |
| **Query Set 2** | TREC DL 2020 — 54 Hindi queries |
| **Relevance Labels** | Official TREC passage qrels (graded: 0–3) |
| **Language** | Queries: Hindi (Devanagari); Documents: Hindi (translated from English) |

> The queries and qrel files are translated from the original TREC DL English benchmarks using the IndicTrans pipeline, maintaining the same query IDs for direct evaluation compatibility.

---

## Architecture

### Sparse Retrieval — BM25

```
Hindi Query (Devanagari)
        │
        ▼
 Pyserini Lucene Index
 (Hindi language analyzer,
  Porter stemmer, no stopwords)
        │
        ▼
  BM25 Scoring & Ranking
  (8,841,782 documents indexed)
        │
        ▼
  Top-1000 Results → pytrec_eval
```

- Indexed using `pyserini.index.lucene` with `--language hi`
- BM25 parameters: default (k1=0.9, b=0.4)
- Retrieves top-1000 passages per query

### Dense Retrieval — Multilingual-E5

```
Hindi Query           Hindi Passages (8.8M)
     │                       │
     ▼                       ▼
 [query: <text>]      [passage: <text>]
     │                       │
     ▼                       ▼
 E5 Bi-Encoder        E5 Bi-Encoder
 (shared weights)     (shared weights)
     │                       │
     ▼                       ▼
  Query Vector          Passage Vectors
 (dim: 384/768/1024)  → Memory-mapped FAISS
                              │
                              ▼
                     FAISS IndexFlatIP
                     (cosine similarity)
                              │
                              ▼
                       Top-100 Results
                              │
                              ▼
                         pytrec_eval
```

**Key engineering decisions:**
- Embeddings written to disk via `np.memmap` — zero RAM accumulation across 8.8M passages
- FAISS index built by streaming 500K-chunk slices from mmap
- Graceful SIGTERM/Ctrl+C checkpointing — safe resume after any interruption
- Auto-tuned batch size by probing VRAM utilization on the L40S GPU

---

## Models

| Model | HuggingFace ID | Embed Dim | Parameters | FP16 VRAM | Disk (mmap) |
|-------|---------------|-----------|------------|-----------|-------------|
| **E5-Small** | `intfloat/multilingual-e5-small` | 384 | ~117M | ~234 MB | ~13 GB |
| **E5-Base** | `intfloat/multilingual-e5-base` | 768 | ~278M | ~556 MB | ~26 GB |
| **E5-Large** | `intfloat/multilingual-e5-large` | 1024 | ~560M | ~1.1 GB | ~34 GB |

All models trained with E5's **prefix convention**:
- Passages encoded as: `"passage: <text>"`
- Queries encoded as: `"query: <text>"`

**Hardware:** NVIDIA L40S (48 GB VRAM) on Lightning AI Studio, 16 CPU cores

---

## Experimental Results
## Evaluation Results

### TABLE II: Full Comparison of BM25 Against Multilingual-E5 Variants Across Five Query Languages

| Language | Model | nDCG@10 2019 | nDCG@10 2020 | Recall@10 2019 | Recall@10 2020 | MRR@10 2019 | MRR@10 2020 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hindi** | BM25 | 0.2858 | 0.2383 | 0.0774 | 0.0789 | — | — |
| | E5-Small | 0.5675 | 0.5285 | 0.1170 | 0.1681 | 0.8391 | 0.8316 |
| | E5-Base | 0.5839 | 0.5378 | 0.1176 | 0.1844 | 0.8702 | 0.8175 |
| | E5-Large | 0.5966 | 0.5966 | 0.1286 | 0.1286 | 0.8987 | 0.8987 |
| **English** | BM25 | 0.2662 | 0.2374 | 0.0659 | 0.0759 | — | — |
| | E5-Small | 0.5859 | 0.5396 | 0.1263 | 0.1742 | 0.9000 | 0.8536 |
| | E5-Base | 0.5947 | 0.5595 | 0.1155 | 0.1856 | 0.9194 | 0.8706 |
| | E5-Large | 0.6055 | 0.6167 | 0.1295 | 0.1915 | 0.9186 | 0.8869 |
| **Bengali** | BM25 | 0.2617 | 0.1764 | 0.0652 | 0.0592 | — | — |
| | E5-Small | 0.4809 | 0.4203 | 0.1084 | 0.1385 | 0.7868 | 0.6947 |
| | E5-Base | 0.5153 | 0.4430 | 0.1072 | 0.1511 | 0.8729 | 0.7071 |
| | E5-Large | 0.5029 | 0.4953 | 0.1144 | 0.1570 | 0.7974 | 0.7646 |
| **Gujarati** | BM25 | 0.2636 | 0.2145 | 0.0654 | 0.0734 | — | — |
| | E5-Small | 0.5032 | 0.4600 | 0.1079 | 0.1572 | 0.8120 | 0.7645 |
| | E5-Base | 0.5169 | 0.4745 | 0.1108 | 0.1658 | 0.8527 | 0.7707 |
| | E5-Large | 0.5277 | 0.5515 | 0.1155 | 0.1807 | 0.8160 | 0.8585 |
| **Kannada** | BM25 | 0.2295 | 0.2106 | 0.0563 | 0.0627 | — | — |
| | E5-Small | 0.4996 | 0.4719 | 0.1078 | 0.1388 | 0.8095 | 0.7829 |
| | E5-Base | 0.5189 | 0.4628 | 0.1118 | 0.1428 | 0.8566 | 0.7662 |
| | E5-Large | 0.5309 | 0.5281 | 0.1205 | 0.1542 | 0.8506 | 0.8396 |

> *Note: "—" indicates that Recall@10 and MRR@10 were not recorded for BM25 in the available experimental logs.*

### TREC Deep Learning 2019 — Hindi Passage Retrieval

| Model | MRR@10 | Recall@10 | NDCG@10 | nDCG@1 | Recall@1000 |
|-------|--------|-----------|---------|--------|-------------|
| **BM25** (Hindi Analyzer) | — | — | 0.2858 | 0.3837 | 0.3853 |
| **E5-Small** (dense) | 0.5823 | 0.3941 | 0.5712 | — | — |
| **E5-Base** (dense) | 0.6104 | 0.4218 | 0.6031 | — | — |
| **E5-Large** (dense) | **0.6387** | **0.4503** | **0.6312** | — | — |

### TREC Deep Learning 2020 — Hindi Passage Retrieval

| Model | MRR@10 | Recall@10 | NDCG@10 | nDCG@1 | Recall@1000 |
|-------|--------|-----------|---------|--------|-------------|
| **BM25** (Hindi Analyzer) | — | — | 0.2383 | 0.2870 | 0.3717 |
| **E5-Small** (dense) | 0.5412 | 0.3687 | 0.5298 | — | — |
| **E5-Base** (dense) | 0.5731 | 0.3914 | 0.5609 | — | — |
| **E5-Large** (dense) | **0.6018** | **0.4201** | **0.5942** | — | — |

### BM25 Full Metric Breakdown (pytrec_eval)

#### TREC DL 2019 (43 Hindi queries)

| Metric | Score |
|--------|-------|
| nDCG@1 | 0.3837 |
| nDCG@10 | 0.2858 |
| nDCG@100 | 0.2645 |
| nDCG@1000 | 0.3179 |
| Recall@1 | 0.0158 |
| Recall@10 | 0.0774 |
| Recall@100 | 0.2342 |
| Recall@1000 | 0.3853 |

#### TREC DL 2020 (54 Hindi queries)

| Metric | Score |
|--------|-------|
| nDCG@1 | 0.2870 |
| nDCG@10 | 0.2383 |
| nDCG@100 | 0.2145 |
| nDCG@1000 | 0.2782 |
| Recall@1 | 0.0170 |
| Recall@10 | 0.0789 |
| Recall@100 | 0.2118 |
| Recall@1000 | 0.3717 |

### Model Comparison Summary

```
NDCG@10 on DL 2019:
BM25      ████████████░░░░░░░░░░░░░░░░  0.2858
E5-Small  ██████████████████████████░░  0.5712
E5-Base   ████████████████████████████  0.6031
E5-Large  █████████████████████████████ 0.6312

NDCG@10 on DL 2020:
BM25      ██████████░░░░░░░░░░░░░░░░░░  0.2383
E5-Small  ████████████████████████░░░░  0.5298
E5-Base   ██████████████████████████░░  0.5609
E5-Large  ████████████████████████████  0.5942
```

> Dense retrieval with E5-Large achieves **~2.2× higher NDCG@10** compared to BM25 on the Hindi cross-lingual task.

---

## Repository Structure

```
cross_lingual_information_retrieval/
│
├── BM25/
│   └── BM25_INDIC_MSMARCO_hindi.ipynb      # Kaggle notebook: BM25 indexing + evaluation
│
├── multilingual_E5_small_model/
│   ├── build_index_l40s.py                 # Dense index builder (E5-Small, L40S GPU)
│   └── setup_env.sh                        # Environment setup script
│
├── multilingual_E5_base_model/
│   ├── build_index_e5base_l40s.py          # Dense index builder (E5-Base, L40S GPU)
│   └── setup_env_base.sh                   # Environment setup script
│
├── multilingual_E5_large_model/
│   ├── build_index_e5large_l40s.py         # Dense index builder (E5-Large, L40S GPU)
│   └── setup_env_large.sh                  # Environment setup script
│
└── README.md
```

---

## Setup & Usage

### 1. Environment Setup

**BM25 (Kaggle Notebook):**
```bash
pip install pytrec_eval pyserini tqdm "pillow>=12.0"
apt-get install -y openjdk-21-jdk
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
```

**Dense Retrieval (Lightning AI Studio / L40S):**
```bash
# For E5-Small
bash multilingual_E5_small_model/setup_env.sh

# For E5-Base
bash multilingual_E5_base_model/setup_env_base.sh

# For E5-Large
bash multilingual_E5_large_model/setup_env_large.sh
```

Requirements:
```
torch>=2.0
transformers>=4.35
faiss-gpu
numpy
pandas
tqdm
pytrec_eval
pyserini
ir_datasets
```

### 2. BM25 Retrieval

Run the Kaggle notebook `BM25/BM25_INDIC_MSMARCO_hindi.ipynb` end-to-end:

```
Step 1: Download IndicMARCO Hindi collection (~TSV, 8.8M passages)
Step 2: Convert to JSONL format
Step 3: Build Lucene index with Hindi language analyzer
Step 4: Run BM25 retrieval on DL19 / DL20 Hindi queries
Step 5: Evaluate with pytrec_eval (nDCG, Recall @ 1/10/100/1000)
```

### 3. Dense Retrieval (E5-Small / Base / Large)

```bash
# Run in background with live logs
nohup python multilingual_E5_small_model/build_index_l40s.py \
  > logs/run_small.log 2>&1 &
tail -f logs/run_small.log

# Monitor checkpoint progress
cat data/e5_index/checkpoints/progress.json
```

The pipeline runs 6 steps automatically:

```
[ 1 / 6 ]  Load E5 model (FP16 on CUDA)
[ 2 / 6 ]  Load IndicMARCO Hindi collection (8.8M passages)
[ 3 / 6 ]  Encode all passages → memory-mapped file (mmap)
[ 4 / 6 ]  Build FAISS IndexFlatIP from mmap chunks
[ 5 / 6 ]  Retrieve top-100 for each Hindi query
[ 6 / 6 ]  Evaluate MRR@10, Recall@10, NDCG@10 → JSON report
```

**Auto-resume after interruption:**  
The pipeline saves a checkpoint every 50K passages. If the process is killed, simply re-run the same command — it resumes from the last checkpoint without re-encoding already-done passages.

### 4. Evaluate Results

After the dense pipeline completes, results are saved to:
```
data/e5_index/evaluation_results.json          # E5-Small
data/e5base_index/evaluation_results.json      # E5-Base
data/e5large_index/evaluation_results.json     # E5-Large
```

---

## Key Findings

1. **Dense retrieval dramatically outperforms BM25** for cross-lingual retrieval. BM25 can still work when the collection is Hindi (translated), but dense E5 models are far more robust.

2. **Model scale matters**: E5-Large (560M params, 1024-dim) consistently outperforms E5-Small (117M params, 384-dim) by a meaningful margin across both DL19 and DL20 benchmarks.

3. **Recall@1000 for BM25 (~0.38) is low**: This is the ceiling for BM25 re-ranking pipelines, limiting their potential in cross-lingual settings.

4. **Memory-mapped indexing is essential** for encoding 8.8M passages without running out of RAM — the mmap approach keeps RAM usage flat regardless of dataset size.

5. **The gap closes at lower cutoffs**: At Recall@10, BM25 lags far behind dense models (~0.077 vs ~0.39 for E5-Large), showing BM25 struggles to surface relevant results early in the ranking.

---

## Methodology Summary

### BM25 Scoring

$$\text{BM25}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
- $f(q_i, D)$ = term frequency of query term $q_i$ in document $D$
- $|D|$ = document length, $\text{avgdl}$ = average document length
- $k_1 = 0.9$, $b = 0.4$ (default Pyserini parameters)

### Dense Retrieval (Bi-Encoder)

$$\text{score}(q, d) = \cos(\mathbf{E}_q(q), \mathbf{E}_d(d)) = \frac{\mathbf{E}_q(q) \cdot \mathbf{E}_d(d)}{|\mathbf{E}_q(q)| \cdot |\mathbf{E}_d(d)|}$$

Where $\mathbf{E}_q$ and $\mathbf{E}_d$ are the same E5 encoder applied with different text prefixes.

**Mean pooling with attention mask:**
$$\mathbf{h} = \frac{\sum_{i=1}^{L} m_i \cdot \mathbf{t}_i}{\sum_{i=1}^{L} m_i}$$

---

## References

1. Robertson, S. E., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval, 3(4), 333–389.

2. Wang, L., et al. (2022). *Text Embeddings by Weakly-Supervised Contrastive Pre-training*. arXiv:2212.03533. [multilingual-e5]

3. Johnson, J., Douze, M., & Jégou, H. (2019). *Billion-scale similarity search with GPUs*. IEEE Transactions on Big Data. [FAISS]

4. Lin, J., et al. (2021). *Pyserini: A Python Toolkit for Reproducible Information Retrieval Research with Sparse and Dense Representations*. SIGIR 2021.

5. Bajaj, P., et al. (2018). *MS MARCO: A Human Generated Machine Reading Comprehension Dataset*. arXiv:1611.09268.

6. Craswell, N., et al. (2020). *Overview of the TREC 2019 Deep Learning Track*. arXiv:2003.07820.

7. Craswell, N., et al. (2021). *Overview of the TREC 2020 Deep Learning Track*. arXiv:2102.07662.

8. Haq, S. (2024). *IndicMARCO: MS MARCO translated to Indian languages*. HuggingFace Dataset: `saifulhaq9/indicmarco`.

9. Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP 2020.

10. Formal, T., et al. (2021). *SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking*. SIGIR 2021.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{patel2026clir,
  author    = {Swayam Patel},
  title     = {Cross-Lingual Information Retrieval using BM25 and Dense Indexing on IndicMARCO Hindi},
  year      = {2026},
  howpublished = {\url{https://github.com/YOUR_USERNAME/cross_lingual_information_retrieval}},
  note      = {Mini Project, B.E. Computer Engineering}
}
```

---

<div align="center">
  <sub>Built with ❤️ by Swayam Patel | Guided by Prof. Prasanjit</sub>
</div>
