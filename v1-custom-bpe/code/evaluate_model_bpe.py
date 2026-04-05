import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import ByteLevelBPETokenizer
import json
import os
import sys
import argparse
from collections import Counter
import re

# Force UTF-8 for Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Model definition
batch_size = 64
block_size = 512
n_embd = 256
n_head = 8
n_layer = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class VedLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, None

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, eos_token=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if eos_token is not None and idx_next.item() == eos_token:
                break
        return idx

# --- Metrics ---

def get_ngram_counts(text, n):
    ngrams = []
    for i in range(len(text) - n + 1):
        ngrams.append(text[i:i+n])
    return Counter(ngrams)

def ngram_overlap_frequency(gen_counts, train_counts):
    if not gen_counts:
        return 0.0, 0.0
    type_overlap = len(set(gen_counts.keys()) & set(train_counts.keys())) / max(1, len(gen_counts))
    shared_count = sum(min(gen_counts[k], train_counts.get(k, 0)) for k in gen_counts)
    token_overlap = shared_count / max(1, sum(gen_counts.values()))
    return type_overlap, token_overlap

def count_syllables(text):
    independent_vowels = len(re.findall(r'[\u0904-\u0914]', text))
    vowel_matras = len(re.findall(r'[\u093E-\u094C]', text))
    consonants = re.findall(r'[\u0915-\u0939]', text)
    viramas = len(re.findall(r'\u094D', text))
    matras = vowel_matras
    inherent = max(0, len(consonants) - viramas - matras)
    return independent_vowels + vowel_matras + inherent

def near_duplicate_check(gen_lines, train_text, min_length=15):
    exact = 0
    near = 0
    total = 0
    for line in gen_lines:
        line = line.strip()
        if len(line) < min_length:
            continue
        total += 1
        if line in train_text:
            exact += 1
        else:
            found_near = False
            step = max(1, len(line) // 4)
            for i in range(0, len(line) - min_length + 1, step):
                if line[i:i+min_length] in train_text:
                    found_near = True
                    break
            if found_near:
                near += 1
    return exact, near, total

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Ved Model BPE")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='accented', choices=['accented', 'plain'])
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to generate for evaluation')
    parser.add_argument('--tokens_per_sample', type=int, default=400, help='Tokens per sample')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    args = parser.parse_args()

    # Resolve paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tokenizer_dir = os.path.join(base_dir, "data", "bpe_tokenizer")
    train_path = os.path.join(base_dir, "data", f"veda_train_{args.dataset}.txt")

    if not os.path.exists(tokenizer_dir):
        print(f"Error: Tokenizer not found at {tokenizer_dir}")
        sys.exit(1)

    tokenizer = ByteLevelBPETokenizer(
        os.path.join(tokenizer_dir, "vocab.json"),
        os.path.join(tokenizer_dir, "merges.txt")
    )
    vocab_size = tokenizer.get_vocab_size()
    eos_token = tokenizer.token_to_id("<eos>")

    with open(train_path, 'r', encoding='utf-8') as f:
        train_text = f.read()

    print("Computing training n-gram distributions...", flush=True)
    train_bigrams = get_ngram_counts(train_text, 2)
    train_trigrams = get_ngram_counts(train_text, 3)

    # Init model
    model = VedLanguageModel(vocab_size)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    print(f"=== Evaluating Checkpoint: {args.checkpoint} ===", flush=True)
    print(f"Settings: temperature={args.temperature}, top_k={args.top_k}, samples={args.num_samples}\n", flush=True)

    # Generate multiple samples
    all_bigram_type, all_bigram_token = [], []
    all_trigram_type, all_trigram_token = [], []
    all_syllable_counts = []
    all_exact = 0
    all_near = 0
    all_total_lines = 0
    all_texts = []

    for sample_idx in range(args.num_samples):
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        gen_tokens = model.generate(context, max_new_tokens=args.tokens_per_sample,
                                     temperature=args.temperature, top_k=args.top_k, eos_token=eos_token)[0].tolist()
        gen_text = tokenizer.decode(gen_tokens)
        all_texts.append(gen_text)

        gen_bigrams = get_ngram_counts(gen_text, 2)
        gen_trigrams = get_ngram_counts(gen_text, 3)

        bi_type, bi_token = ngram_overlap_frequency(gen_bigrams, train_bigrams)
        tri_type, tri_token = ngram_overlap_frequency(gen_trigrams, train_trigrams)
        all_bigram_type.append(bi_type)
        all_bigram_token.append(bi_token)
        all_trigram_type.append(tri_type)
        all_trigram_token.append(tri_token)

        lines = gen_text.split('\n')
        valid_lines = [l for l in lines if len(l.strip()) > 5]
        syllable_counts = [count_syllables(l) for l in valid_lines[:10]]
        all_syllable_counts.extend(syllable_counts)

        exact, near, total = near_duplicate_check(valid_lines, train_text)
        all_exact += exact
        all_near += near
        all_total_lines += total

    # --- Report ---
    print("=" * 60, flush=True)
    print("[N-GRAM ANALYSIS] (averaged over {} samples)".format(args.num_samples), flush=True)
    print(f"  Bigram  — Type overlap: {sum(all_bigram_type)/len(all_bigram_type)*100:.1f}%  |  Token overlap: {sum(all_bigram_token)/len(all_bigram_token)*100:.1f}%", flush=True)
    print(f"  Trigram — Type overlap: {sum(all_trigram_type)/len(all_trigram_type)*100:.1f}%  |  Token overlap: {sum(all_trigram_token)/len(all_trigram_token)*100:.1f}%", flush=True)

    print(f"\n[METER & SYLLABLE ANALYSIS]", flush=True)
    print(f"  Syllables per line (first 10): {all_syllable_counts[:10]}", flush=True)
    if all_syllable_counts:
        avg_syl = sum(all_syllable_counts) / len(all_syllable_counts)
        std_syl = (sum((s - avg_syl)**2 for s in all_syllable_counts) / len(all_syllable_counts)) ** 0.5
        print(f"  Average: {avg_syl:.1f}  |  Std Dev: {std_syl:.1f}", flush=True)
        print(f"  Consistency: {'GOOD - low variance' if std_syl < 3 else 'WEAK - high variance'}", flush=True)

    print(f"\n[MEMORIZATION CHECK]", flush=True)
    print(f"  Exact copies:     {all_exact} / {all_total_lines} lines", flush=True)
    print(f"  Near-duplicates:  {all_near} / {all_total_lines} lines", flush=True)
    if all_total_lines > 0:
        mem_pct = (all_exact + all_near) / all_total_lines * 100
        if mem_pct < 10:
            verdict = "GOOD - model is generating, not copying"
        elif mem_pct < 30:
            verdict = "WARNING - some memorization detected"
        else:
            verdict = "OVERFIT - heavy memorization"
        print(f"  Verdict: {verdict} ({mem_pct:.0f}%)", flush=True)

    print(f"\n[SAMPLE GENERATION] (Sample 1 of {args.num_samples})", flush=True)
    print("-" * 60, flush=True)
    print(all_texts[0], flush=True)
    print("-" * 60, flush=True)
