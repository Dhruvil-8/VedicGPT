import argparse
import json
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_dataset
from tokenizers import Tokenizer
from transformers import (
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from train_vedic_unigram_tokenizer import train_tokenizer


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FILE = BASE_DIR / "data" / "veda_train_accented.txt"
DEFAULT_TOKENIZER_DIR = BASE_DIR / "tokenizer"
DEFAULT_MODEL_DIR = BASE_DIR / "model"
DEFAULT_CHECKPOINT_DIR = DEFAULT_MODEL_DIR / "checkpoints"
DEFAULT_FINAL_MODEL_DIR = DEFAULT_MODEL_DIR / "vedic-unigram-gpt"
DEFAULT_LOG_FILE = DEFAULT_MODEL_DIR / "training_log_vedic.txt"
DEFAULT_RUN_CONFIG = DEFAULT_MODEL_DIR / "training_run_config.json"

STYLE_PROMPTS = ["<RIG> ", "<YAJUR> ", "<ATHARVA> "]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Vedic GPT model with a Unigram tokenizer.")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--final-model-dir", type=Path, default=DEFAULT_FINAL_MODEL_DIR)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument("--run-config-file", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--test-size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--augment-factor", type=int, default=2)
    parser.add_argument("--train-tokenizer-if-missing", action="store_true")
    return parser.parse_args()


class SampleLogCallback(TrainerCallback):
    def __init__(self, tokenizer: PreTrainedTokenizerFast, log_file: Path, device: str, context_length: int):
        self.tokenizer = tokenizer
        self.log_file = log_file
        self.device = device
        self.context_length = context_length

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs or "loss" not in logs:
            return

        model = kwargs["model"]
        model.eval()
        samples: list[str] = []
        for prompt in STYLE_PROMPTS:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output_ids = model.generate(
                    inputs.input_ids,
                    max_length=min(self.context_length, 96),
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            samples.append(self.tokenizer.decode(output_ids[0], skip_special_tokens=False))
        model.train()

        lines = [
            f"step={state.global_step}",
            f"loss={logs['loss']:.4f}",
            f"eval_loss={logs['eval_loss']:.4f}" if "eval_loss" in logs else "eval_loss=NA",
            f"lr={logs.get('learning_rate', 0):.2e}",
            f"RIG={samples[0]}",
            f"YAJUR={samples[1]}",
            f"ATHARVA={samples[2]}",
            "=" * 60,
        ]
        entry = "\n".join(lines) + "\n"
        print(entry)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(entry)


def load_fast_tokenizer(tokenizer_dir: Path) -> PreTrainedTokenizerFast:
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if not tokenizer_json.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_json}")

    tokenizer_obj = Tokenizer.from_file(str(tokenizer_json))
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        bos_token="<s>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )


def ensure_tokenizer(args: argparse.Namespace) -> PreTrainedTokenizerFast:
    tokenizer_json = args.tokenizer_dir / "tokenizer.json"
    if tokenizer_json.exists():
        return load_fast_tokenizer(args.tokenizer_dir)

    if not args.train_tokenizer_if_missing:
        raise FileNotFoundError(
            f"Tokenizer missing at {tokenizer_json}. Run "
            f"`python scripts/train_vedic_unigram_tokenizer.py` first or pass `--train-tokenizer-if-missing`."
        )

    train_tokenizer(
        data_file=args.data_file,
        output_dir=args.tokenizer_dir,
        vocab_size=args.vocab_size,
    )
    return load_fast_tokenizer(args.tokenizer_dir)


def main() -> None:
    args = parse_args()

    if not args.data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found at {args.data_file}. Run preprocess_vedic.py first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = ensure_tokenizer(args)

    dataset = load_dataset("text", data_files={"train": str(args.data_file)})

    def tokenize_batch(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.context_length)

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=["text"])
    split = tokenized["train"].train_test_split(test_size=args.test_size, seed=args.seed)

    train_original = split["train"].shuffle(seed=args.seed)
    if args.augment_factor > 1:
        split["train"] = concatenate_datasets([train_original] * args.augment_factor).shuffle(seed=args.seed)
    else:
        split["train"] = train_original

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=args.context_length,
        n_ctx=args.context_length,
        n_embd=args.embedding_dim,
        n_layer=args.num_layers,
        n_head=args.num_heads,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)

    steps_per_epoch = max(
        1,
        len(split["train"]) // max(1, args.batch_size * args.gradient_accumulation_steps),
    )
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(50, int(total_steps * 0.1))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.final_model_dir.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    args.run_config_file.parent.mkdir(parents=True, exist_ok=True)

    run_config = {
        "device": device,
        "data_file": str(args.data_file),
        "tokenizer_dir": str(args.tokenizer_dir),
        "output_dir": str(args.output_dir),
        "final_model_dir": str(args.final_model_dir),
        "log_file": str(args.log_file),
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "embedding_dim": args.embedding_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "augment_factor": args.augment_factor,
        "train_size": len(split["train"]),
        "eval_size": len(split["test"]),
        "estimated_total_steps": total_steps,
        "warmup_steps": warmup_steps,
    }
    args.run_config_file.write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        do_train=True,
        do_eval=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[
            SampleLogCallback(
                tokenizer=tokenizer,
                log_file=args.log_file,
                device=device,
                context_length=args.context_length,
            ),
            EarlyStoppingCallback(early_stopping_patience=4),
        ],
    )

    print(f"Device: {device}")
    print(f"Dataset: {args.data_file}")
    print(f"Tokenizer: {args.tokenizer_dir / 'tokenizer.json'}")
    print(f"Augmented train size: {len(split['train'])}")
    print(f"Eval size: {len(split['test'])}")
    print(f"Estimated total steps: {total_steps}")
    print(
        "Config tuned for Colab T4 and the current Vedic corpus: "
        "512 ctx, 6-layer GPT, 384 hidden size, fp16 when CUDA is present."
    )

    trainer.train()
    trainer.save_model(str(args.final_model_dir))
    tokenizer.save_pretrained(str(args.final_model_dir))

    print(f"Final model saved to: {args.final_model_dir}")


if __name__ == "__main__":
    main()
