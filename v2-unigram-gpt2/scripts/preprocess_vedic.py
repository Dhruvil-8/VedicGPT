import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = BASE_DIR / "data" / "raw"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data"

ACCENTED_OUTPUT = "veda_train_accented.txt"
REPORT_DIRNAME = "reports"
REPORT_FILE = "preprocess_report.json"
SUSPICIOUS_FILE = "suspicious_lines.txt"

MOJIBAKE_CHUNK_RE = re.compile(r"[\x80-\xff]+")
ACCENT_RE = re.compile(r"[\u0951\u0952\u1CD0-\u1CFF\uA8E0-\uA8FF]")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F\uA8E0-\uA8FF]")
LATIN_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"[०-९0-9]")
VERSE_SPLIT_RE = re.compile(r"॥\s*[०-९0-9]+\s*॥|॥")
ONLY_COUNTER_RE = re.compile(r"^[\s०-९0-9,./()\-–—]+$")
SPACE_RE = re.compile(r"\s+")

CORPORA = [
    ("Rigveda", "<RIG>"),
    ("Yajurveda", "<YAJUR>"),
    ("AtharvaVeda", "<ATHARVA>"),
]

METER_KEYWORDS = {
    "गायत्री",
    "अनुष्टुप्",
    "त्रिष्टुप्",
    "जगती",
    "विराट्",
    "विराड्",
    "पङ्क्ति",
    "पङ्क्तिः",
    "बृहती",
    "उष्णिक्",
    "ककुम्मती",
    "शक्वरी",
    "अतिशक्वरी",
    "महाबृहती",
    "पथ्यापङ्क्तिः",
    "द्विपदा",
    "चतुष्पदा",
    "षट्पदा",
    "एकपदा",
    "त्रिपदा",
    "सप्तपदा",
    "दशपदा",
    "भुरिक्",
    "गर्भा",
    "आर्षी",
    "आसुरी",
    "साम्नी",
    "याजुषी",
    "प्राजापत्या",
    "प्रस्तारपङ्क्तिः",
    "आस्तारपङ्क्तिः",
    "त्र्यवसाना",
    "एकावसाना",
    "पुरउष्णिक्",
    "पुरस्कृति",
}

METADATA_KEYWORDS = METER_KEYWORDS | {
    "ऋषि",
    "ऋषिः",
    "ऋषयः",
    "देवता",
    "देवताः",
    "छन्दः",
    "छन्दांसि",
    "अध्यायः",
    "अध्याय",
    "अथर्वा",
    "अथर्वाचार्यः",
    "मेधाजननम्",
    "रोगोपशमनम्",
    "मूत्रमोचनम्",
    "अपां भेषजम्",
    "यातुधाननाशनम्",
    "विजयाय प्रार्थना",
    "पापविमोचनम्",
    "अध्यात्मम्",
}

CHAR_REPLACEMENTS = str.maketrans(
    {
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",
        "\xa0": " ",
        "–": "-",
        "—": "-",
        "−": "-",
        "‑": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "¦": "।",
        "|": "।",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean Vedic JSON corpus for retraining.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-devanagari", type=int, default=1)
    return parser.parse_args()


def repair_mojibake(text: str, counters: Counter) -> str:
    def _fix(match: re.Match[str]) -> str:
        chunk = match.group(0)
        try:
            fixed = chunk.encode("latin1").decode("utf-8")
        except UnicodeDecodeError:
            return chunk
        if fixed != chunk:
            counters["repaired_chunks"] += 1
        return fixed

    text = text.translate(CHAR_REPLACEMENTS)
    text = MOJIBAKE_CHUNK_RE.sub(_fix, text)
    text = text.replace("।।", "॥")
    text = unicodedata.normalize("NFC", text)
    return text


def normalize_block(text: str, counters: Counter) -> str:
    text = repair_mojibake(text, counters)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"॥\s*॥", "॥", text)
    return text.strip()


def has_metadata_keyword(text: str) -> bool:
    return any(keyword in text for keyword in METADATA_KEYWORDS)


def cleanup_clause_text(text: str) -> str:
    text = re.sub(r"[(){}\[\],;:\"'!?-]", " ", text)
    text = DIGIT_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def looks_like_metadata_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if ONLY_COUNTER_RE.match(stripped):
        return True

    clean = cleanup_clause_text(stripped.replace("।", " ").replace("॥", " "))
    if not clean:
        return True

    accent_count = len(ACCENT_RE.findall(stripped))
    clauses = [cleanup_clause_text(part) for part in re.split(r"[।॥]", stripped) if cleanup_clause_text(part)]
    short_clause_count = sum(1 for clause in clauses if len(clause) <= 32)

    if has_metadata_keyword(stripped) and accent_count == 0:
        return True
    if len(clauses) >= 2 and short_clause_count == len(clauses) and accent_count == 0:
        return True
    if DIGIT_RE.search(stripped) and accent_count == 0 and len(clean) <= 140:
        return True
    if stripped.count(",") >= 2 and accent_count == 0:
        return True
    return False


def looks_like_metadata_clause(clause: str) -> bool:
    stripped = clause.strip()
    if not stripped:
        return True

    clean = cleanup_clause_text(stripped)
    if not clean:
        return True
    if ACCENT_RE.search(stripped):
        return False
    if has_metadata_keyword(stripped):
        return True
    if DIGIT_RE.search(stripped):
        return True
    if len(clean.split()) <= 4 and len(clean) <= 36:
        return True
    return False


def strip_leading_metadata(text: str) -> str:
    text = SPACE_RE.sub(" ", text).strip()
    if not text:
        return ""

    clauses = [part.strip() for part in text.split("।")]
    while clauses and looks_like_metadata_clause(clauses[0]):
        clauses.pop(0)
    while clauses and not clauses[-1]:
        clauses.pop()
    return "। ".join(clauses).strip()


def finalize_verse_text(text: str, min_devanagari: int) -> str:
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\{[^}]*\}", " ", text)
    text = DIGIT_RE.sub(" ", text)
    text = LATIN_RE.sub(" ", text)
    text = re.sub(r"[,:;\"!?]", " ", text)
    text = text.replace("।", " । ")
    text = text.replace("॥", " ॥ ")
    text = SPACE_RE.sub(" ", text).strip(" ।")

    if len(DEVANAGARI_RE.findall(text)) < min_devanagari:
        return ""
    if not text:
        return ""
    if not text.endswith("॥"):
        if text.endswith("।"):
            text = text[:-1].rstrip() + " ॥"
        else:
            text = text + " ॥"
    return SPACE_RE.sub(" ", text).strip()


def split_into_segments(text: str) -> list[str]:
    return [segment.strip() for segment in VERSE_SPLIT_RE.split(text) if segment.strip()]


def extract_verses(raw_text: str, tag: str, min_devanagari: int, counters: Counter) -> list[str]:
    normalized = normalize_block(raw_text, counters)
    segments = split_into_segments(normalized)
    verses: list[str] = []

    for segment in segments:
        lines = [line.strip() for line in segment.splitlines() if line.strip()]
        content_lines = [line for line in lines if not looks_like_metadata_line(line)]
        if not content_lines:
            content_lines = lines
            counters["metadata_fallback_segments"] += 1

        merged = " ".join(content_lines)
        original_merged = merged
        merged = strip_leading_metadata(merged)
        if not merged:
            merged = original_merged
        merged = finalize_verse_text(merged, min_devanagari=min_devanagari)
        if not merged:
            counters["dropped_segments"] += 1
            continue

        verses.append(f"{tag} {merged} <eos>")

    return verses


def build_corpus(raw_dir: Path, min_devanagari: int) -> tuple[list[str], dict]:
    accented_lines: list[str] = []
    seen_accented: set[str] = set()

    report: dict[str, object] = {
        "raw_dir": str(raw_dir),
        "counts": {},
        "repaired_chunks": 0,
        "dropped_segments": 0,
        "duplicate_accented_lines": 0,
    }

    suspicious: list[str] = []

    for folder_name, tag in CORPORA:
        corpus_dir = raw_dir / folder_name
        corpus_counter = Counter()
        file_count = 0
        item_count = 0
        output_count = 0

        if not corpus_dir.exists():
            report["counts"][folder_name] = {
                "files": 0,
                "items": 0,
                "verses": 0,
                "warning": "missing directory",
            }
            continue

        for file_path in sorted(corpus_dir.glob("*.json")):
            file_count += 1
            with file_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)

            if not isinstance(data, list):
                continue

            for item in data:
                item_count += 1
                text = item.get("text", "")
                for verse in extract_verses(text, tag, min_devanagari=min_devanagari, counters=corpus_counter):
                    output_count += 1
                    if verse in seen_accented:
                        report["duplicate_accented_lines"] += 1
                    else:
                        accented_lines.append(verse)
                        seen_accented.add(verse)

        report["counts"][folder_name] = {
            "files": file_count,
            "items": item_count,
            "verses": output_count,
            "repaired_chunks": corpus_counter["repaired_chunks"],
            "dropped_segments": corpus_counter["dropped_segments"],
            "metadata_fallback_segments": corpus_counter["metadata_fallback_segments"],
        }
        report["repaired_chunks"] += corpus_counter["repaired_chunks"]
        report["dropped_segments"] += corpus_counter["dropped_segments"]

    suspicious_markers = sorted(METADATA_KEYWORDS)
    for line in accented_lines:
        content = re.sub(r"^<[^>]+>\s+", "", line)
        content = content.replace("<eos>", "").strip()
        if LATIN_RE.search(content):
            suspicious.append(line)
            continue
        if any(marker in content for marker in suspicious_markers):
            suspicious.append(line)
            continue
        if DIGIT_RE.search(content):
            suspicious.append(line)

    report["final"] = {
        "accented_lines": len(accented_lines),
        "suspicious_lines": len(suspicious),
    }
    return accented_lines, {"report": report, "suspicious": suspicious}


def write_outputs(output_dir: Path, accented_lines: list[str], extras: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / REPORT_DIRNAME
    reports_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / ACCENTED_OUTPUT).write_text("\n".join(accented_lines) + "\n", encoding="utf-8")
    (reports_dir / REPORT_FILE).write_text(
        json.dumps(extras["report"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports_dir / SUSPICIOUS_FILE).write_text(
        "\n".join(extras["suspicious"]) + ("\n" if extras["suspicious"] else ""),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    accented_lines, extras = build_corpus(
        raw_dir=args.raw_dir,
        min_devanagari=args.min_devanagari,
    )
    write_outputs(args.output_dir, accented_lines, extras)

    report = extras["report"]
    print(f"Raw corpus: {args.raw_dir}")
    print(f"Accented output: {args.output_dir / ACCENTED_OUTPUT}")
    print(f"Suspicious lines: {report['final']['suspicious_lines']}")
    print(f"Repaired mojibake chunks: {report['repaired_chunks']}")
    print(f"Dropped noisy segments: {report['dropped_segments']}")
    print(f"Final accented lines: {report['final']['accented_lines']}")


if __name__ == "__main__":
    main()
