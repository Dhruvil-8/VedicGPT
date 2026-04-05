---
title: VedicGPT Explorer
emoji: 🔱
colorFrom: yellow
colorTo: green
sdk: gradio
python_version: 3.11
app_file: app.py
pinned: false
license: apache-2.0
---

# VedicGPT: Byte-Pair Encoding (BPE) Transformer for Vedic Sanskrit

**VedicGPT** is an experimental transformer-based language model trained from scratch on a consolidated corpus of the Rigveda, Yajurveda, and Atharvaveda. The project explores how much structural and phonetic information can be learned from a relatively small Vedic Sanskrit dataset using modern NLP techniques.

> ⚠️ **Important Disclaimer**
> This project is a computational experiment. The generated text:
>
> * is **not authentic Vedic scripture**
> * should **not be used for ritual, recitation, or scholarly interpretation**
> * does **not preserve semantic or oral tradition accuracy**
>
> The model learns statistical patterns, not meaning or tradition.

---

## 🎯 Objectives

* Explore **low-resource language modeling** on Vedic Sanskrit (~4.5MB corpus)
* Analyze whether transformers can learn:
  * phonetic structure
  * verse boundaries (`।`, `॥`)
  * stylistic variation across Vedas
* Evaluate limitations of:
  * character-level vs subword tokenization
  * small-scale transformer architectures

---

## ✨ Features

* **Accent-Preserving BPE Tokenizer**
  Handles Devanagari Unicode carefully to retain Vedic accent markers (Udatta `॑`, Anudatta `॒`, etc.)

* **Conditioned Generation**
  Uses control tokens:
  ```text
  <RIG>  <YAJUR>  <ATHARVA>
  ```
  to model stylistic variation across Vedic corpora

* **Structured Output Learning**
  Incorporates `॥ <eos>` to model verse boundaries and termination

* **Evaluation Framework**
  * N-gram overlap (phonetic similarity)
  * Memorization detection
  * Syllable-length consistency (approximate meter)

* **Colab T4 Optimized**
  Designed to train efficiently on free-tier GPU (~10k steps ≈ 1–2 hours)

---

## 🧠 Model Details

| Property | Value |
| :--- | :--- |
| Architecture | Transformer (decoder-only) |
| Layers | 6 |
| Attention Heads | 8 |
| Embedding Dim | 256 |
| Parameters | ~5.89M |
| Tokenizer | BPE (≈2,000 vocab) |

---

## 📁 Repository Structure

```
Veda/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── code/
│   ├── prepare_corpus.py         # Data cleaning & merging pipeline
│   ├── train_tokenizer.py        # BPE tokenizer training
│   ├── train_veda_bpe.py         # Transformer training script
│   ├── evaluate_model_bpe.py     # Evaluation: n-gram, meter, memorization
│   └── veda_bpe_colab.ipynb      # Google Colab notebook (T4 GPU)
│
└── data/
    ├── raw/                      # Original JSON sources (Rigveda, Yajurveda, Atharvaveda)
    ├── veda_train_accented.txt   # Processed training corpus (with accents)
    ├── veda_train_plain.txt      # Processed training corpus (plain)
    ├── bpe_tokenizer/            # BPE vocab.json + merges.txt
    └── final_bpe_model_export/   # Trained model weights (model_weights.pt)
```

---

## 🚀 Getting Started

### Prerequisites
* Python 3.9+
* CUDA GPU (or Google Colab T4)

### Installation

```bash
git clone <repo-url>
cd Veda
pip install -r requirements.txt
```

---

### Step 1: Prepare the Corpus

```bash
python code/prepare_corpus.py
```

Downloads and cleans the Vedic texts into `data/veda_train_accented.txt`.

---

### Step 2: Train the BPE Tokenizer

```bash
python code/train_tokenizer.py
```

Trains a Byte-Pair Encoding tokenizer on the accented corpus and saves to `data/bpe_tokenizer/` (vocab size: 2,000).

---

### Step 3: Train the Model

```bash
python code/train_veda_bpe.py --dataset accented --max_iters 10000
```

---

### Step 4: Evaluate

```bash
python code/evaluate_model_bpe.py \
  --checkpoint data/final_bpe_model_export/model_weights.pt \
  --num_samples 5
```

---

### Google Colab (Free T4 GPU)

1. Upload the `code/` and `data/` folders to your Google Drive.
2. Open `code/veda_bpe_colab.ipynb` in Colab.
3. Set Runtime → **T4 GPU**.
4. Run all cells — training completes in ~1–2 hours.

---

## ⚠️ Limitations

This model:

* does **not understand Sanskrit semantics**
* does **not learn true Vedic chandas (meter rules)**
* may exhibit **memorization** due to small dataset size
* generates **synthetic, approximate text**, not authoritative content

---

## 🙏 Ethical Note

This project intended solely for:

* computational experimentation
* linguistic pattern analysis
* educational exploration

It does **not attempt to replicate, replace, or reinterpret** sacred knowledge systems.

---

## 📚 Data Source

Dataset consolidated from:

👉 **[DharmicData by bhavykhatri](https://github.com/bhavykhatri/DharmicData)**

---

