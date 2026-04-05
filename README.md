# SanskritGPT-Vedic

**An AI-Assisted Computational Experiment in Vedic Sanskrit Generation**

This repository documents the iterative development of a language model for generating **accent-annotated Vedic Sanskrit** verse. The project is a research experiment exploring how modern neural architectures can learn the phonological and prosodic patterns of an ancient, highly-inflected language — including its distinctive pitch accent system (Udatta and Anudatta).

> **Disclaimer**: This is an AI-assisted training experiment. Generated text is statistically derived from the corpus and does not represent authentic Vedic scripture, scholarly translation, or traditional recitation. It must not be used for religious, ritual, or canonical academic purposes.

---

## Project Structure

The repository is organized into two versions, each representing a significant evolution in architecture and tokenization strategy:

### v1-custom-bpe

| Attribute | Detail |
|:---|:---|
| **Architecture** | 6-layer Decoder-only Transformer (Custom PyTorch) |
| **Tokenizer** | Byte-Pair Encoding (BPE), 2,000 Vocabulary |
| **Parameters** | ~5.8 Million |
| **Status** | Legacy / Baseline |

Initial experiment establishing the training pipeline, data preprocessing, and baseline generation quality on the Vedic corpus.

### v2-unigram-gpt2

| Attribute | Detail |
|:---|:---|
| **Architecture** | GPT-2 (Hugging Face Transformers) |
| **Tokenizer** | Unigram + Metaspace (PreTrainedTokenizerFast), 8,000 Vocabulary |
| **Parameters** | ~10.1 Million |
| **Status** | Current / Active |

Upgraded experiment with a more expressive tokenizer that natively preserves Devanagari Unicode and Vedic accent characters. Introduces style-control tokens (`<RIG>`, `<YAJUR>`, `<ATHARVA>`) for Veda-specific generation.

---

## Live Resources

| Resource | Version | Link |
|:---|:---|:---|
| **Hugging Face Model** | v2 (Unigram) | [Dhruvil8/SanskritGPT-Vedic](https://huggingface.co/Dhruvil8/SanskritGPT-Vedic) |
| **Interactive Space** | v2 (Unigram) | [spaces/Dhruvil8/SanskritGPT-Vedic](https://huggingface.co/spaces/Dhruvil8/SanskritGPT-Vedic) |
| **Related Epic Model** | Itihasa (42M) | [Dhruvil8/SanskritGPT-Itihasa](https://huggingface.co/Dhruvil8/SanskritGPT-Itihasa) |

---

## Evaluation Results (v2)

The v2 model is evaluated on its ability to replicate the statistical distribution of Vedic pitch accents — a more meaningful metric than perplexity alone for this domain.

| Metric | Value |
|:---|:---|
| **Evaluation Loss** | 6.5157 |
| **Perplexity** | ~675.69 |
| **Udatta (॑) — Ground Truth** | 6.56% |
| **Udatta (॑) — Generated** | 6.62% |
| **Anudatta (॒) — Ground Truth** | 8.69% |
| **Anudatta (॒) — Generated** | 8.39% |

Both Udatta and Anudatta accent density deviations are within **1% of ground truth**, indicating the model has learned the prosodic fingerprint of the Vedic corpus.

---

## Scope and Limitations

- **Statistical learning only**: The model identifies patterns in pitch accent placement, syllabic structure, and inter-Veda stylistic variation. It does not possess linguistic or semantic understanding.
- **Small corpus**: The Vedic corpus is limited by modern NLP standards, which constrains generalization.
- **High perplexity**: Expected given the 8K vocabulary and the extreme morphological richness of Vedic Sanskrit. Accent density fidelity is the primary quality indicator.
- **Not trained on Samaveda**: The current experiment covers the Rigveda, Yajurveda, and Atharvaveda only.

---

## Credits

- **Dataset**: [DharmicData by bhavykhatri](https://github.com/bhavykhatri/DharmicData)
- **Framework**: PyTorch + Hugging Face Transformers
- **License**: MIT — underlying Vedic texts are in the public domain.
