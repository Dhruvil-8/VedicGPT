import os
import json
import re
import unicodedata

# Resolve paths relative to the script location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_RAW = os.path.join(BASE_DIR, "data", "raw")

DATA_DIR_RIG = os.path.join(DATA_DIR_RAW, "Rigveda")
DATA_DIR_YAJUR = os.path.join(DATA_DIR_RAW, "Yajurveda")
DATA_DIR_ATHARVA = os.path.join(DATA_DIR_RAW, "AtharvaVeda")

DATA_DIRS = [
    (DATA_DIR_RIG, "<RIG>"),
    (DATA_DIR_YAJUR, "<YAJUR>"),
    (DATA_DIR_ATHARVA, "<ATHARVA>")
]

OUTPUT_DIR = os.path.join(BASE_DIR, "data")
ACCENTED_FILE = os.path.join(OUTPUT_DIR, "veda_train_accented.txt")
PLAIN_FILE = os.path.join(OUTPUT_DIR, "veda_train_plain.txt")
VOCAB_ACCENTED_FILE = os.path.join(OUTPUT_DIR, "vocab_accented.json")
VOCAB_PLAIN_FILE = os.path.join(OUTPUT_DIR, "vocab_plain.json")

# Regex to remove numbers, Latin characters, and common punctuation non-native to the verses
# NOTE: Danda (।) and Double Danda (॥) are KEPT as structural signals
CLEAN_REGEX = re.compile(r'[०-९0-9\(\)\-\.,;A-Za-z]')

# Regex to remove Vedic accents to create the "plain" version
# \u0951 is Udatta (॑)
# \u0952 is Anudatta (॒)
# \u1CD0-\u1CFF are Vedic Extensions (various tone marks)
# \uA8E0-\uA8FF are Devanagari Extended (cantillation marks)
ACCENT_REGEX = re.compile(r'[\u0951\u0952\u1CD0-\u1CFF\uA8E0-\uA8FF]')

def process_corpus():
    accented_lines = []
    plain_lines = []
    
    for data_dir, tag in DATA_DIRS:
        if not os.path.exists(data_dir):
            print(f"Directory not found: {data_dir}")
            continue
            
        for filename in os.listdir(data_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        for item in data:
                            if 'text' in item:
                                text_block = item['text']
                                # Unicode NFC normalization to prevent silent vocab splits
                                text_block = unicodedata.normalize("NFC", text_block)
                                # Split into individual lines
                                for line in text_block.split('\n'):
                                    # Skip metadata lines
                                    metadata_keywords = [':', 'गायत्री', 'त्रिष्टुप्', 'अनुष्टुप्', 'जगती', 'विराज्', 'पङ्क्ति', 'मधुच्छन्दा', 'वैश्वामित्र', 'ऋषि']
                                    if any(x in line for x in metadata_keywords):
                                        continue
                                    
                                    # 1. Replace full verse markers like ॥१॥ with a single ॥ signal
                                    line = re.sub(r'॥[०-९\d]+॥', '॥', line)
                                    
                                    # 2. Apply general cleaning (numbers, latin, etc.)
                                    cleaned = CLEAN_REGEX.sub('', line)
                                    
                                    # 3. Normalize dandas and spaces
                                    cleaned = cleaned.replace("॥॥", "॥")
                                    cleaned = cleaned.replace("।।", "।")
                                    
                                    # 4. Add spacing around dandas for better boundary learning
                                    cleaned = cleaned.replace("।", " । ")
                                    cleaned = cleaned.replace("॥", " ॥ ")
                                    
                                    # 5. Final whitespace normalization
                                    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                                    
                                    if len(cleaned) > 5:  # Skip empty or very short lines
                                        # Prepend the Veda tag
                                        accented_verse = tag + " " + cleaned
                                        
                                        # Only add EOS marker after double danda (end of verse)
                                        if cleaned.endswith("॥"):
                                            accented_verse = accented_verse + " <eos>"
                                        
                                        accented_lines.append(accented_verse)
                                        
                                        # Create plain version
                                        plain = ACCENT_REGEX.sub('', cleaned)
                                        plain_verse = tag + " " + plain
                                        if plain.endswith("॥"):
                                            plain_verse = plain_verse + " <eos>"
                                        plain_lines.append(plain_verse)
                    except json.JSONDecodeError:
                        print(f"Error reading JSON: {filepath}")

    # Remove strict duplicates while maintaining order
    def unique_lines(lines):
        seen = set()
        res = []
        for l in lines:
            if l not in seen:
                seen.add(l)
                res.append(l)
        return res

    accented_lines = unique_lines(accented_lines)
    plain_lines = unique_lines(plain_lines)

    print(f"Total unique accented lines: {len(accented_lines)}")
    print(f"Total unique plain lines: {len(plain_lines)}")

    # Write text files
    with open(ACCENTED_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(accented_lines))
        
    with open(PLAIN_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(plain_lines))
        
    # Build and save Character Vocabularies
    def save_vocab(text_content, vocab_filename):
        chars = sorted(list(set(text_content)))
        char2idx = {ch: i for i, ch in enumerate(chars)}
        idx2char = {i: ch for i, ch in enumerate(chars)}
        vocab_data = {
            "vocab_size": len(chars),
            "char2idx": char2idx,
            "idx2char": idx2char
        }
        with open(vocab_filename, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        return len(chars)

    accented_vocab_size = save_vocab('\n'.join(accented_lines), VOCAB_ACCENTED_FILE)
    plain_vocab_size = save_vocab('\n'.join(plain_lines), VOCAB_PLAIN_FILE)

    print(f"Accented Vocab Size: {accented_vocab_size}")
    print(f"Plain Vocab Size: {plain_vocab_size}")
    print("Corpus prepared successfully.")

if __name__ == "__main__":
    process_corpus()
