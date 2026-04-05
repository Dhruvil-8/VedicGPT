import json
import os
import re
from pathlib import Path
import unicodedata

# Paths
BASE_DIR = Path(r"C:\Users\admin\Downloads\VedSastra")
RAW_DATA_DIR = BASE_DIR / "Veda" / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "Veda" / "data" # Save output to Veda/data/ to be safe
OUTPUT_FILE = OUTPUT_DIR / "veda_train_accented.txt"

# Metadata keywords to filter out (at start of mantras)
METADATA_KEYWORDS = [
    "ऋषिः", "देवता", "छन्द", "गायत्री", "अनुष्टुप्", "त्रिष्टुप्", "जगती", "विराट्", "पङ्क्ति", "बृहती", "उष्णिग्", "द्विपदा", "चतुष्पदा", "षट्पदा",
    "मधुच्छन्दा", "वैश्वामित्र", "अथर्वा", "पर्जन्य", "मेधाजननम्", "रोगोपशमनम्", "मूत्रमोचनम्", "अपां भेषजम्", "यातुधाननाशनम्", "विजयाय प्रार्थना", "पाप -विमोचनम्"
]

def clean_text(text):
    if not text: return ""
    
    # Normalize Unicode NFC
    text = unicodedata.normalize('NFC', text)
    
    # Remove lines or fragments that are clearly metadata or headers
    lines = text.strip().split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Skip if line is just a number or common header
        if re.match(r'^[०-९0-9\s\.\-–]+$', line):
            continue
            
        # Skip if line contains multiple metadata words
        hits = sum(1 for kw in METADATA_KEYWORDS if kw in line)
        if hits >= 1 and len(line) < 60: # Metadata lines are usually short
            continue
            
        # Strip potential manual verse counters like "१-४ " at the very start
        line = re.sub(r'^[०-९0-9\s\.\-–]{1,10}', '', line).strip()
        
        if line:
            cleaned_lines.append(line)
    
    full_text = " ".join(cleaned_lines)
    
    # 1. Remove elaborate verse numbering patterns: 
    # ॥१॥, ॥१॥ १, ॥२२॥॥, (५८), (१) ९, [१] etc.
    full_text = re.sub(r'॥[०-९0-9\s\.\-\(\)\[\]]*॥', ' ', full_text)
    full_text = re.sub(r'॥[०-९0-9\s\.\-\(\)\[\]]*', ' ', full_text) 
    
    # 2. Remove specific parenthetical noise (१) [१]
    full_text = re.sub(r'[\(\[\{][०-९0-9\s\-\.]*[\)\]\}]', ' ', full_text)
    
    # 3. Strip all remaining digits (Devanagari and Western)
    full_text = re.sub(r'[०-९0-9]+', ' ', full_text)
    
    # 4. Remove standalone dandas as we will add a double danda at the end
    full_text = full_text.replace('॥', ' ').replace('।', ' ')
    
    # Normalize whitespace
    full_text = re.sub(r'\s+', ' ', full_text).strip()
    
    return full_text

def process_rigveda():
    results = []
    rigveda_dir = RAW_DATA_DIR / "Rigveda"
    if not rigveda_dir.exists(): return []
    
    for file_path in rigveda_dir.glob("*.json"):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                text = item.get("text", "")
                if "॥" in text:
                    segments = re.split(r'॥[०-९0-9\s\.\-\(\)]*॥|॥', text)
                else:
                    segments = [text]
                    
                for s in segments:
                    cleaned = clean_text(s)
                    if cleaned and len(cleaned) > 10:
                        results.append(f"<RIG> {cleaned} ॥ <eos>")
    return results

def process_yajurveda():
    results = []
    yajur_dir = RAW_DATA_DIR / "Yajurveda"
    if not yajur_dir.exists(): return []
    
    for file_path in yajur_dir.glob("*.json"):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                text = item.get("text", "")
                segments = re.split(r'॥[०-९0-9\s\.\-]*॥|॥', text)
                for s in segments:
                    cleaned = clean_text(s)
                    if cleaned and len(cleaned) > 10:
                        results.append(f"<YAJUR> {cleaned} ॥ <eos>")
    return results

def process_atharvaveda():
    results = []
    atharva_dir = RAW_DATA_DIR / "Atharvaveda"
    if not atharva_dir.exists(): return []
    
    for file_path in atharva_dir.glob("*.json"):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                text = item.get("text", "")
                segments = re.split(r'॥[०-९0-9\s\.\-]*॥|॥', text)
                for s in segments:
                    cleaned = clean_text(s)
                    if cleaned and len(cleaned) > 10:
                        results.append(f"<ATHARVA> {cleaned} ॥ <eos>")
    return results

def main():
    print(f"Pre-processing Vedic Corpus from: {RAW_DATA_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_lines = []
    
    rv_lines = process_rigveda()
    all_lines.extend(rv_lines)
    print(f"Rigveda processed: {len(rv_lines)} verses")
    
    y_lines = process_yajurveda()
    all_lines.extend(y_lines)
    print(f"Yajurveda processed: {len(y_lines)} verses")
    
    a_lines = process_atharvaveda()
    all_lines.extend(a_lines)
    print(f"Atharvaveda processed: {len(a_lines)} verses")
    
    seen = set()
    unique_lines = []
    for line in all_lines:
        if line not in seen:
            unique_lines.append(line)
            seen.add(line)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for line in unique_lines:
            f.write(line + "\n")
    
    print(f"Extraction complete. Total unique verses: {len(unique_lines)}")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
