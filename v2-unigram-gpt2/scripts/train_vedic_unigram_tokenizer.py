import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Metaspace
from transformers import PreTrainedTokenizerFast


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FILE = BASE_DIR / "data" / "veda_train_accented.txt"
DEFAULT_OUTPUT_DIR = BASE_DIR / "tokenizer"

SPECIAL_TOKENS = [
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
    "<RIG>",
    "<YAJUR>",
    "<ATHARVA>",
    "<eos>",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Unigram tokenizer for the cleaned Vedic corpus.")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--min-lines", type=int, default=100)
    return parser.parse_args()


def load_sample_line(data_file: Path) -> str:
    for line in data_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return line.strip()
    return ""


def load_training_lines(data_file: Path) -> list[str]:
    return [line.strip() for line in data_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_initial_alphabet(data_file: Path) -> list[str]:
    text = data_file.read_text(encoding="utf-8")
    return sorted({char for char in text if not char.isspace()})


def train_tokenizer(data_file: Path, output_dir: Path, vocab_size: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    training_lines = load_training_lines(data_file)
    initial_alphabet = build_initial_alphabet(data_file)

    tokenizer_obj = Tokenizer(models.Unigram())
    tokenizer_obj.normalizer = normalizers.Sequence([NFKC()])
    tokenizer_obj.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="always")
    tokenizer_obj.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        unk_token="<unk>",
        initial_alphabet=initial_alphabet,
    )
    tokenizer_obj.train_from_iterator(training_lines, trainer=trainer, length=len(training_lines))
    tokenizer_obj.save(str(output_dir / "tokenizer.json"))

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        bos_token="<s>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    fast_tokenizer.save_pretrained(str(output_dir))

    sample_line = load_sample_line(data_file)
    sample_encoding = fast_tokenizer(sample_line) if sample_line else None

    stats = {
        "data_file": str(data_file),
        "output_dir": str(output_dir),
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": len(fast_tokenizer),
        "special_tokens": SPECIAL_TOKENS,
        "training_line_count": len(training_lines),
        "initial_alphabet_size": len(initial_alphabet),
        "sample_line": sample_line,
        "sample_token_count": len(sample_encoding["input_ids"]) if sample_encoding else 0,
    }
    (output_dir / "tokenizer_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return stats


def main() -> None:
    args = parse_args()
    if not args.data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found at {args.data_file}. Run preprocess_vedic.py first."
        )

    line_count = sum(1 for line in args.data_file.read_text(encoding="utf-8").splitlines() if line.strip())
    if line_count < args.min_lines:
        raise ValueError(
            f"Dataset at {args.data_file} has only {line_count} non-empty lines; expected at least {args.min_lines}."
        )

    stats = train_tokenizer(
        data_file=args.data_file,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
    )

    print(f"Tokenizer dataset: {args.data_file}")
    print(f"Tokenizer output: {args.output_dir}")
    print(f"Requested vocab size: {stats['requested_vocab_size']}")
    print(f"Actual vocab size: {stats['actual_vocab_size']}")
    print(f"Sample token count: {stats['sample_token_count']}")


if __name__ == "__main__":
    main()
