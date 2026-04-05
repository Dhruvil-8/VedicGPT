from tokenizers import ByteLevelBPETokenizer
import os
import sys

# Force UTF-8 for Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# --- Config ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_file = os.path.join(base_dir, "data", "veda_train_accented.txt")
output_dir = os.path.join(base_dir, "data", "bpe_tokenizer")
vocab_size = 2000
# --------------

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Training BPE tokenizer on {train_file}...")
tokenizer = ByteLevelBPETokenizer()

# Train with special tokens
tokenizer.train(
    files=[train_file],
    vocab_size=vocab_size,
    min_frequency=2,
    show_progress=True,
    special_tokens=["<eos>", "<RIG>", "<YAJUR>", "<ATHARVA>"]
)

# Save the model
tokenizer.save_model(output_dir)
print(f"Tokenizer saved to {output_dir}")

# Test the tokenizer
test_str = "अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म् ।"
encoded = tokenizer.encode(test_str)
print(f"\nTest String: {test_str}")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")
decoded = tokenizer.decode(encoded.ids)
print(f"Decoded: {decoded}")
