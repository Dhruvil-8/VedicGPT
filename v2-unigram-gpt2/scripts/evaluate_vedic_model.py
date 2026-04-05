import argparse
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

# Fix for Windows console character encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments
)

# Constants for Vedic Sanskrit
UDATTA = "॑"  # U+0951
ANUDATTA = "॒"  # U+0952
VEDIC_TAGS = ["<RIG>", "<YAJUR>", "<ATHARVA>", "<SAM>"]

def parse_args():
    parser = argparse.ArgumentParser(description="Research-Grade Evaluation for Vedic Accented Model")
    parser.add_argument("--model_path", type=str, default="model/model_output_accented", help="Path to saved model and tokenizer")
    parser.add_argument("--data_path", type=str, default="data/veda_train_accented.txt", help="Path to evaluation dataset")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples per branch")
    parser.add_argument("--sample_size", type=int, default=1000, help="Number of verses to use for statistical evaluation")
    parser.add_argument("--max_len", type=int, default=100, help="Max length for generation")
    parser.add_argument("--output_report", type=str, default="model_evaluation_report.txt", help="Path to save evaluation results")
    return parser.parse_args()

def calculate_perplexity(model, tokenizer, data_file, device, sample_size=1000):
    """Calculates perplexity on a subset of the dataset for speed."""
    dataset = load_dataset("text", data_files={"test": data_file})["test"]
    
    # Shuffle and subset for CPU-bound hardware
    if len(dataset) > sample_size:
        dataset = dataset.shuffle(seed=42).select(range(sample_size))
        print(f"Subsetting to {sample_size} verses for high-speed CPU evaluation.")
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    training_args = TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=1,  # Low RAM safety
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        eval_dataset=tokenized
    )
    
    eval_results = trainer.evaluate()
    perplexity = math.exp(eval_results["eval_loss"])
    return perplexity, eval_results["eval_loss"]

def audit_accent_density(text):
    """Audits the frequency of Vedic accents compared to Devanagari characters."""
    total_chars = len(re.sub(r'\s+', '', text))
    udattas = text.count(UDATTA)
    anudattas = text.count(ANUDATTA)
    
    stats = {
        "total_chars": total_chars,
        "udattas": udattas,
        "anudattas": anudattas,
        "udatta_ratio": (udattas / total_chars) * 100 if total_chars > 0 else 0,
        "anudatta_ratio": (anudattas / total_chars) * 100 if total_chars > 0 else 0
    }
    return stats

def generate_samples(model, tokenizer, device, num_samples, max_len):
    """Generates and audits samples for each Vedic branch."""
    results = {}
    for tag in VEDIC_TAGS:
        prompt = f"{tag} "
        samples = []
        for _ in range(num_samples):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    inputs.input_ids,
                    max_length=max_len,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )
            decoded = tokenizer.decode(output_ids[0], skip_special_tokens=False)
            samples.append(decoded)
        results[tag] = samples
    return results

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model and tokenizer from {args.model_path}...")
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path).to(device)
    model.eval()

    print(f"Step 1: Calculating Perplexity on {args.sample_size} samples...")
    ppl, loss = calculate_perplexity(model, tokenizer, args.data_path, device, args.sample_size)
    
    print("Step 2: Performing Accent Density Audit...")
    # Reference audit on ground truth (first 1000 lines)
    with open(args.data_path, "r", encoding="utf-8") as f:
        ground_truth_sample = "".join(f.readlines()[:1000])
    truth_stats = audit_accent_density(ground_truth_sample)

    print("Step 3: Generating Model Samples & Consistency Check...")
    branch_samples = generate_samples(model, tokenizer, device, args.num_samples, args.max_len)
    
    # Analyze generated samples
    all_generated = "".join([s for tag in branch_samples for s in branch_samples[tag]])
    gen_stats = audit_accent_density(all_generated)

    # Generate Report
    report = f"""--- VEDIC MODEL RESEARCH-GRADE EVALUATION REPORT ---
Model Path: {args.model_path}
Dataset Evaluation: {args.data_path}
Device: {device}

1. QUANTITATIVE ANALYSIS
- Evaluation Loss: {loss:.4f}
- Perplexity (PPL): {ppl:.4f}

2. LINGUISTIC AUDIT (ACCENT DENSITY)
Metric                Ground Truth (%)    Model Generated (%)
-----------------------------------------------------------
Udatta Ratio (॑)      {truth_stats['udatta_ratio']:.2f}%               {gen_stats['udatta_ratio']:.2f}%
Anudatta Ratio (॒)    {truth_stats['anudatta_ratio']:.2f}%               {gen_stats['anudatta_ratio']:.2f}%

(Comparison Note: A difference < 1% indicates high-fidelity learning of Vedic prosody.)

3. BRANCH STYLE SAMPLES
"""
    for tag in VEDIC_TAGS:
        report += f"\n--- {tag} Samples ---\n"
        for i, s in enumerate(branch_samples[tag]):
            report += f"[{i+1}] {s}\n"

    report += "\n--- END OF REPORT ---\n"
    
    print(report)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"Full evaluation report saved to: {args.output_report}")

if __name__ == "__main__":
    main()
