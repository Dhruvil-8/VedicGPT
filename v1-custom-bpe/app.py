import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import ByteLevelBPETokenizer
import gradio as gr
import os
import re

# --- Model Architecture (Must match exactly) ---
block_size = 512
n_embd = 256
n_head = 8
n_layer = 6
dropout = 0.2

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

    def forward(self, idx, targets=None):
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
            logits, _ = self(idx_cond)
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

# --- Global Config (Adjusted for Hugging Face Space) ---
device = 'cpu'
# Model and Tokenizer should be in the 'model/' directory on the Space
tokenizer_dir = "model/tokenizer"
model_path = "model/model_weights.pt"

# --- Load Resources ---
# Fallback for local testing if model/ doesn't exist yet
if not os.path.exists(model_path):
    # Try the local machine path as a fallback
    local_model = "data/final_bpe_model_export-20260327T083241Z-3-001/final_bpe_model_export/model_weights.pt"
    local_tok = "data/bpe_tokenizer"
    if os.path.exists(local_model):
        model_path = local_model
        tokenizer_dir = local_tok
tokenizer = ByteLevelBPETokenizer(
    os.path.join(tokenizer_dir, "vocab.json"),
    os.path.join(tokenizer_dir, "merges.txt")
)
vocab_size = tokenizer.get_vocab_size()
eos_token_id = tokenizer.token_to_id("<eos>")

model = VedLanguageModel(vocab_size)
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# --- Helpers ---
def highlight_verse(text):
    # Highlight verse boundaries and control tokens
    text = text.replace("<RIG>", "[RIG]")
    text = text.replace("<YAJUR>", "[YAJUR]")
    text = text.replace("<ATHARVA>", "[ATHARVA]")
    text = text.replace("॥", "॥")
    text = text.replace("।", "।")
    text = text.replace("<eos>", "[EOS]")
    return text

def check_style_strength(text):
    # Heuristic for style strength based on common Vedic patterns
    if "अ॒ग्नि" in text or "इन्द्र॑" in text:
        return "HIGH (Strong Rigvedic patterns detected)"
    if "॥" in text and "।" in text:
        return "MEDIUM (Proper verse structure)"
    return "LOW (Experimental/Abstract output)"

def check_memorization(generated):
    # Very basic check against a few common verses
    known_verses = [
        "अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म्",
        "इषे॑ त्वो॒र्जे त्वा॑ वा॒यवः॑ स्थ दे॒वो वः॑ सवि॒ता",
        "यज्ञस्य॑ घा॒रेष्विन्द्र॑"
    ]
    for verse in known_verses:
        if verse in generated:
            return "⚠️ Warning: Potential memorization of training data detected."
    return "✅ Output appears to be novel composition."

REAL_VERSES = {
    "<RIG>": "अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म् । होता॑रं रत्न॒धात॑मम् ॥",
    "<YAJUR>": "इषे॑ त्वो॒र्जे त्वा॑ वा॒यवः॑ स्थ दे॒वो वः॑ सवि॒ता प्राप॑यतु श्रेष्ठ॑तमाय॒ कर्म॑णे ॥",
    "<ATHARVA>": "ये त्रि॑ष॒प्ताः प॑रि॒यन्ति॒ विश्वा॑ रू॒पाणि॑ बिभ्र॑तः । वा॒चस्पति॒र्बला॑ तेषां त॒न्वो॑ अ॒द्य द॑धातु मे ॥"
}

# --- UI Logic ---
def vedic_generator(veda_style, temp, top_k, max_tokens, lock_style):
    start_token = veda_style
    context = torch.tensor([tokenizer.encode(start_token).ids], dtype=torch.long, device=device)
    
    # Generate variations
    variations = []
    for _ in range(3):
        out_ids = model.generate(context, max_new_tokens=max_tokens, temperature=temp, top_k=top_k, eos_token=eos_token_id)[0].tolist()
        gen_text = tokenizer.decode(out_ids)
        variations.append(gen_text)
    
    # Analysis
    primary_output = variations[0]
    highlighted = highlight_verse(primary_output)
    style_strength = check_style_strength(primary_output)
    mem_check = check_memorization(primary_output)
    syllables = len(re.findall(r'[अ-ह]़?्?ा?ि?ी?ु?ू?ृ?ॄ?े?ै?ो?ौ?ं?ः?', primary_output))
    
    real_comparison = REAL_VERSES.get(veda_style, "No sample available.")
    
    return highlighted, variations[1], variations[2], style_strength, mem_check, syllables, real_comparison

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 🔱 VedicGPT Explorer")
    gr.Markdown("### Explore Vedic Sanskrit Patterns with Transformer-based AI")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Generation Controls")
            veda_style = gr.Dropdown(choices=["<RIG>", "<YAJUR>", "<ATHARVA>"], value="<RIG>", label="Select Veda Style")
            gr.Markdown("*Rigveda: Hymns | Yajurveda: Ritual | Atharvaveda: Practical*")
            
            temp = gr.Slider(0.1, 2.0, value=0.8, label="Temperature (Creativity)")
            top_k = gr.Slider(10, 200, value=50, step=1, label="Top-k (Diversity)")
            max_tokens = gr.Slider(10, 300, value=128, step=1, label="Max New Tokens")
            lock_style = gr.Checkbox(label="Lock Style (Force Control Token)", value=True)
            
            gen_btn = gr.Button("🔥 Generate Vedic Verses", variant="primary")
            
            gr.Markdown("---")
            gr.Markdown("### 📚 Click-to-Load Prompts")
            gr.Examples(
                examples=[["<RIG>", 0.7, 50, 100, True], ["<YAJUR>", 0.8, 80, 100, True], ["<ATHARVA>", 0.9, 100, 100, True]],
                inputs=[veda_style, temp, top_k, max_tokens, lock_style]
            )

        with gr.Column(scale=2):
            gr.Markdown("### 🔮 Primary Generated Verse")
            output_main = gr.Textbox(label="Main Output (with analysis tags)", lines=4)
            
            with gr.Row():
                output_v1 = gr.Textbox(label="Variation 2", lines=2)
                output_v2 = gr.Textbox(label="Variation 3", lines=2)
            
            with gr.Row():
                style_ind = gr.Label(label="Style Strength")
                mem_ind = gr.Label(label="Memorization Check")
                syll_count = gr.Number(label="Approx Syllable Count")

            gr.Markdown("---")
            with gr.Accordion("🏛️ Real vs Generated Comparison", open=False):
                gr.Markdown("Compare the model output with a real verse from the selected Veda.")
                real_comp_text = gr.Textbox(label="Real Reference Verse", interactive=False)

    gr.Markdown("---")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🧠 Model Info")
            gr.Markdown("""
            - **Architecture:** 6-layer Decoder-only Transformer
            - **Parameters:** ~5.89 Million
            - **Dataset:** ~4.5MB consolidated Vedic corpus
            - **Tokenizer:** BPE (Byte-Pair Encoding), 2,000 vocab
            """)
        with gr.Column():
            gr.Markdown("### 🛡️ Ethical Disclaimer")
            gr.Markdown("""
            > ⚠️ This project is a computational experiment.
            > - Generated text is **not authentic scripture**.
            > - Not for ritual, recitation, or scholarly study.
            > - The model learns statistical patterns, not meaning.
            """)

    gen_btn.click(
        fn=vedic_generator,
        inputs=[veda_style, temp, top_k, max_tokens, lock_style],
        outputs=[output_main, output_v1, output_v2, style_ind, mem_ind, syll_count, real_comp_text]
    )

    gr.Markdown("#### Footer: [GitHub](https://github.com/Dhruvil-8/SanskritGPT-Vedic)")


if __name__ == "__main__":
    demo.launch()
