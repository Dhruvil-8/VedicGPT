import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import os
import sys
import argparse
import math
# sys.stdout.reconfigure(encoding='utf-8')

parser = argparse.ArgumentParser(description="Train Ved Model")
parser.add_argument('--dataset', type=str, default="accented", choices=["accented", "plain"], help='Which dataset to run on')
parser.add_argument('--max_iters', type=int, default=5000, help='Number of training iterations')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--eval_interval', type=int, default=200, help='How often to evaluate and save checkpoints')
parser.add_argument('--lr', type=float, default=3e-4, help='Max learning rate')
args = parser.parse_args()

# --- Hyperparameters ---
batch_size = args.batch_size
block_size = 256   # ~8 padas config
max_iters = args.max_iters
eval_interval = args.eval_interval
learning_rate = args.lr
min_lr = learning_rate / 10
warmup_iters = 100
lr_decay_iters = max_iters # decay down to min_lr at the end
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
n_embd = 192
n_head = 6
n_layer = 4
dropout = 0.2
weight_decay = 0.1
# -----------------------

print(f"Running on device: {device}", flush=True)

# Load dataset and vocab
print("Loading dataset and vocab...", flush=True)
# Base directory of the project
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
text_path = os.path.join(base_dir, "data", f"veda_train_{args.dataset}.txt")
vocab_path = os.path.join(base_dir, "data", f"vocab_{args.dataset}.json")

with open(text_path, 'r', encoding='utf-8') as f:
    text = f.read()

with open(vocab_path, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)

vocab_size = vocab_data["vocab_size"]
char2idx = vocab_data["char2idx"]
# Convert keys back to int
idx2char = {int(k): v for k, v in vocab_data["idx2char"].items()}

encode = lambda s: [char2idx[c] for c in s if c in char2idx]
decode = lambda l: ''.join([idx2char[i] for i in l])

print("Encoding text to tokens...", flush=True)
data = torch.tensor(encode(text), dtype=torch.long)
print(f"Dataset size: {len(data)} tokens", flush=True)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# --- Model Architecture ---

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
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, eos_token=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            # Stop if we hit EOS (if defined)
            if eos_token is not None and idx_next.item() == eos_token:
                break
        return idx
# -----------------------

# --- Training ---

print("Initializing model...", flush=True)
model = VedLanguageModel().to(device)
print(f"Total model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

checkpoint_dir = os.path.join(base_dir, "code", "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
log_file = os.path.join(base_dir, "code", f"training_log_{args.dataset}.txt")

# Find EOS token character (usually '>') if using <|endoftext|>
eos_char = ">"
eos_token = char2idx.get(eos_char, None)

print("Starting training...", flush=True)
for iter in range(max_iters):
    
    # Determine learning rate for this step
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Eval and Save
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        msg = f"step {iter}: lr {lr:.6f}, train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        print(msg, flush=True)
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"ved_{args.dataset}_{iter}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        
        # Briefly test generation
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        gen_tokens = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=20, eos_token=eos_token)[0].tolist()
        gen_text = decode(gen_tokens).replace('\n', ' / ')
        print(f"--- Sample ---", flush=True)
        print(gen_text, flush=True)
        print("--------------", flush=True)
        
        with open(log_file, 'a', encoding='utf-8') as lf:
            lf.write(f"{msg}\nSample: {gen_text}\n\n")

    # Train Step
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

print("Training finished.")
