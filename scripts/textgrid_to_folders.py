"""
build_dataset.py
================
Builds a speechocean762-compatible dataset from annotated TextGrid files.

Filename convention:  spk01_audio0001_scored.TextGrid
  → utt_id  : spk01_audio0001
  → spk_id  : spk01  (everything before first underscore)

Gender / age come from speakers.csv (auto-generated on first run — fill it in, then re-run).

Output layout:
    dataset/
    ├── speakers.csv          ← fill in gender/age, then re-run
    ├── resource/
    │   ├── lexicon.txt
    │   ├── phones.txt
    │   ├── scores.json
    │   └── text-phone
    └── data/
        ├── train/
        │   ├── wav.scp
        │   ├── text
        │   ├── utt2spk
        │   ├── spk2utt
        │   ├── spk2gender
        │   └── spk2age
        └── test/
            └── ...  (same files)

Usage:
    # First run — generates speakers.csv template, fill in gender/age then re-run
    python build_dataset.py

    # Custom dirs
    python build_dataset.py --textgrid_dir "./annotated" --wav_dir ./wav --output_dir ./dataset

    # Train/test split (text file, one utt_id per line)
    python build_dataset.py --test_list ./test_ids.txt
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# TextGrid parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_textgrid(path: str):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    duration_match = re.search(r'xmax\s*=\s*([0-9.]+)', content)
    duration = float(duration_match.group(1)) if duration_match else 0.0

    tiers = {}
    for block in re.split(r'\s+item\s*\[\d+\]:', content)[1:]:
        name_match = re.search(r'name\s*=\s*"([^"]*)"', block)
        if not name_match:
            continue
        tier_name = name_match.group(1).strip()
        intervals = []
        for ib in re.split(r'\s+intervals\s*\[\d+\]:', block)[1:]:
            xmin = float(re.search(r'xmin\s*=\s*([0-9.eE+\-]+)', ib).group(1))
            xmax = float(re.search(r'xmax\s*=\s*([0-9.eE+\-]+)', ib).group(1))
            text_m = re.search(r'text\s*=\s*"([^"]*)"', ib)
            text = text_m.group(1).strip() if text_m else ""
            intervals.append({"xmin": xmin, "xmax": xmax, "text": text})
        tiers[tier_name] = intervals

    return duration, tiers


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SILENCE = {"", "sp", "sil", "SIL", "<eps>", "spn", "SPN"}


def scored(intervals):
    return [iv for iv in intervals if iv["text"].strip() not in SILENCE]


def safe_float(val, tier, idx):
    try:
        return float(val)
    except (ValueError, TypeError):
        raise ValueError(f"Tier '{tier}' interval {idx}: expected number, got '{val}'")


def safe_int(val, tier, idx):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        raise ValueError(f"Tier '{tier}' interval {idx}: expected number, got '{val}'")


def single_score(intervals, tier_name, as_int=True):
    s = scored(intervals)
    if len(s) != 1:
        raise ValueError(f"Tier '{tier_name}': expected 1 scored interval, got {len(s)}")
    return safe_int(s[0]["text"], tier_name, 0) if as_int else safe_float(s[0]["text"], tier_name, 0)


def utt_id_from_stem(stem: str) -> str:
    """spk01_audio0001_scored → spk01_audio0001"""
    return stem.replace("_scored", "")


def spk_from_utt(utt_id: str) -> str:
    """spk01_audio0001 → spk01"""
    return utt_id.split("_")[0]


def find_wav(utt_id: str, wav_dir: Path) -> str:
    for ext in (".wav", ".WAV", ".flac", ".mp3"):
        matches = list(wav_dir.rglob(utt_id + ext))
        if matches:
            return str(matches[0].resolve())
    return f"MISSING:{wav_dir / (utt_id + '.wav')}"


# ─────────────────────────────────────────────────────────────────────────────
# speakers.csv
# ─────────────────────────────────────────────────────────────────────────────

def load_or_create_speakers_csv(output_dir: Path, all_tg: list) -> dict:
    """
    Returns dict: {spk_id: {"gender": "m/f/?", "age": "?"}}
    Generates a template CSV if it doesn't exist yet.
    """
    csv_path = output_dir / "speakers.csv"
    spk_ids  = sorted({spk_from_utt(utt_id_from_stem(p.stem)) for p in all_tg})

    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["speaker_id", "gender", "age"])
            writer.writeheader()
            for spk in spk_ids:
                writer.writerow({"speaker_id": spk, "gender": "?", "age": "?"})
        print(f"\n⚠  speakers.csv created → {csv_path}", file=sys.stderr)
        print("   Fill in gender (m/f) and age for each speaker, then re-run.\n", file=sys.stderr)

    # Load
    speakers = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            spk = row["speaker_id"].strip()
            speakers[spk] = {
                "gender": row.get("gender", "?").strip(),
                "age":    row.get("age",    "?").strip(),
            }

    # Append any new speakers not yet in CSV
    new_spks = [s for s in spk_ids if s not in speakers]
    if new_spks:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["speaker_id", "gender", "age"])
            for spk in new_spks:
                writer.writerow({"speaker_id": spk, "gender": "?", "age": "?"})
                speakers[spk] = {"gender": "?", "age": "?"}
        print(f"  ⚠  Added {len(new_spks)} new speakers to speakers.csv", file=sys.stderr)

    return speakers


# ─────────────────────────────────────────────────────────────────────────────
# scores.json builder
# ─────────────────────────────────────────────────────────────────────────────

def build_scores_entry(tiers: dict) -> dict:
    required = [
        "sentence_acc", "sentence_completeness", "sentence_fluency", "sentence_prosody",
        "words", "word_acc", "word_stress", "phones", "phone_acc",
    ]
    missing = [t for t in required if t not in tiers]
    if missing:
        raise ValueError(f"Missing tiers: {missing}")

    acc          = single_score(tiers["sentence_acc"],          "sentence_acc")
    completeness = single_score(tiers["sentence_completeness"], "sentence_completeness", as_int=False)
    fluency      = single_score(tiers["sentence_fluency"],      "sentence_fluency")
    prosodic     = single_score(tiers["sentence_prosody"],      "sentence_prosody")
    total        = round((acc + completeness + fluency + prosodic) / 4, 2)

    word_ivs      = scored(tiers["words"])
    word_acc_ivs  = scored(tiers["word_acc"])
    word_str_ivs  = scored(tiers["word_stress"])
    phone_ivs     = scored(tiers["phones"])
    phone_acc_ivs = scored(tiers["phone_acc"])

    n = len(word_ivs)
    for name, ivs in [("word_acc", word_acc_ivs), ("word_stress", word_str_ivs)]:
        if len(ivs) != n:
            raise ValueError(f"Tier '{name}' has {len(ivs)} but 'words' has {n}")

    if len(phone_ivs) != len(phone_acc_ivs):
        raise ValueError(f"'phones' {len(phone_ivs)} vs 'phone_acc' {len(phone_acc_ivs)}")

    phone_data = [
        {"xmin": p["xmin"], "xmax": p["xmax"], "label": p["text"],
         "acc": safe_float(a["text"], "phone_acc", i)}
        for i, (p, a) in enumerate(zip(phone_ivs, phone_acc_ivs))
    ]

    words_out = []
    for i, wiv in enumerate(word_ivs):
        w_phones = [p for p in phone_data
                    if p["xmin"] >= wiv["xmin"] - 1e-6
                    and p["xmax"] <= wiv["xmax"] + 1e-6]
        w_acc    = safe_int(word_acc_ivs[i]["text"], "word_acc",    i)
        w_stress = safe_int(word_str_ivs[i]["text"], "word_stress", i)
        w_total  = round((w_acc + w_stress) / 2, 2)
        words_out.append({
            "text":              wiv["text"].upper(),
            "accuracy":          w_acc,
            "stress":            w_stress,
            "total":             w_total,
            "phones":            [p["label"] for p in w_phones],
            "phones-accuracy":   [p["acc"]   for p in w_phones],
            "mispronunciations": [],
        })

    return {
        "accuracy":     acc,
        "completeness": completeness,
        "fluency":      fluency,
        "prosodic":     prosodic,
        "text":         " ".join(w["text"] for w in words_out),
        "total":        total,
        "words":        words_out,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Lexicon builder  (resource/lexicon.txt)
# ─────────────────────────────────────────────────────────────────────────────

def build_lexicon(all_tg: list) -> dict:
    """
    Returns {WORD: [phone, phone, ...]} by cross-referencing words and phones
    tiers via time boundaries.

    If the same word appears with different phone sequences across recordings,
    the last-seen pronunciation wins.  Change the dict value to a set of tuples
    and write one line per entry if you want all unique pronunciations instead.
    """
    lexicon = {}
    for tg_path in all_tg:
        try:
            _, tiers = parse_textgrid(str(tg_path))
        except Exception:
            continue
        if "words" not in tiers or "phones" not in tiers:
            continue
        word_ivs  = scored(tiers["words"])
        phone_ivs = scored(tiers["phones"])
        for wiv in word_ivs:
            phones = [
                p["text"] for p in phone_ivs
                if p["xmin"] >= wiv["xmin"] - 1e-6
                and p["xmax"] <= wiv["xmax"] + 1e-6
            ]
            if phones:
                lexicon[wiv["text"].upper()] = phones
    return lexicon


# ─────────────────────────────────────────────────────────────────────────────
# text-phone builder  (resource/text-phone)
# ─────────────────────────────────────────────────────────────────────────────

def build_text_phone(all_tg: list) -> list:
    """
    Returns lines of the form:
        utt_id  ph1 ph2 ph3 ...
    covering every utterance across all splits.
    """
    lines = []
    for tg_path in sorted(all_tg):
        utt_id = utt_id_from_stem(tg_path.stem)
        try:
            _, tiers = parse_textgrid(str(tg_path))
        except Exception:
            continue
        if "phones" not in tiers:
            continue
        phones = [p["text"] for p in scored(tiers["phones"])]
        lines.append(f"{utt_id} {' '.join(phones)}")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Kaldi split builder
# ─────────────────────────────────────────────────────────────────────────────

def build_split(files, wav_dir, output_dir, split, speakers):
    split_dir = output_dir / "data" / split
    split_dir.mkdir(parents=True, exist_ok=True)

    wav_scp        = []
    text_lines     = []
    utt2spk_lines  = []
    spk2gender_lines = []
    spk2age_lines  = []
    all_phones     = set()
    missing_wav    = 0
    seen_spks      = set()

    for tg_path in sorted(files):
        utt_id = utt_id_from_stem(tg_path.stem)
        spk_id = spk_from_utt(utt_id)

        try:
            _, tiers = parse_textgrid(str(tg_path))
        except Exception as e:
            print(f"  ✗ parse error {tg_path.name}: {e}", file=sys.stderr)
            continue

        if "words" not in tiers or "phones" not in tiers:
            print(f"  ✗ {tg_path.name}: missing words/phones tier", file=sys.stderr)
            continue

        # wav.scp
        wav_path = find_wav(utt_id, wav_dir)
        if wav_path.startswith("MISSING:"):
            print(f"  ⚠ wav not found: {utt_id}", file=sys.stderr)
            missing_wav += 1
        wav_scp.append(f"{utt_id} {wav_path}")

        # text
        transcript = " ".join(iv["text"].upper() for iv in scored(tiers["words"]))
        text_lines.append(f"{utt_id} {transcript}")

        # utt2spk
        utt2spk_lines.append(f"{utt_id} {spk_id}")

        # spk2gender + spk2age  (once per speaker)
        if spk_id not in seen_spks:
            gender = speakers.get(spk_id, {}).get("gender", "?")
            age    = speakers.get(spk_id, {}).get("age",    "?")
            spk2gender_lines.append(f"{spk_id} {gender}")
            spk2age_lines.append(f"{spk_id} {age}")
            seen_spks.add(spk_id)

        # phones for phones.txt accumulation
        phones_seq = [p["text"] for p in scored(tiers["phones"])]
        all_phones.update(phones_seq)

        print(
            f"  ✓ {utt_id}  spk={spk_id}"
            f"  gender={speakers.get(spk_id, {}).get('gender', '?')}"
            f"  age={speakers.get(spk_id, {}).get('age', '?')}",
            file=sys.stderr,
        )

    def wl(name, lines):
        with open(split_dir / name, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(lines)) + "\n")

    wl("wav.scp",    wav_scp)
    wl("text",       text_lines)
    wl("utt2spk",    utt2spk_lines)
    wl("spk2gender", spk2gender_lines)
    wl("spk2age",    spk2age_lines)

    # spk2utt  (derived from utt2spk)
    spk2utt: dict = {}
    for line in utt2spk_lines:
        uid, spk = line.split(" ", 1)
        spk2utt.setdefault(spk, []).append(uid)
    with open(split_dir / "spk2utt", "w", encoding="utf-8") as f:
        for spk in sorted(spk2utt):
            f.write(f"{spk} {' '.join(sorted(spk2utt[spk]))}\n")

    print(f"  [{split}] {len(wav_scp)} utts, {missing_wav} missing wavs → {split_dir}", file=sys.stderr)
    return all_phones


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build speechocean762-compatible dataset")
    parser.add_argument(
        "--textgrid_dir",
        default="/Users/lyn/Desktop/uni/thesis/dataset_making/5. annotated templates",
        help="Dir of *_scored.TextGrid files",
    )
    parser.add_argument(
        "--wav_dir",
        default="/Users/lyn/Desktop/uni/thesis/dataset_making/2. wav_audio + text",
        help="Root wav directory, searched recursively",
    )
    parser.add_argument("--output_dir", default="./dataset", help="Output root (default: ./dataset)")
    parser.add_argument("--test_list",  default=None,
                        help="Text file listing utt_ids for test split, one per line")
    parser.add_argument("--split",      default="train", choices=["train", "test"],
                        help="Default split when no --test_list given (default: train)")
    args = parser.parse_args()

    tg_dir     = Path(args.textgrid_dir)
    wav_dir    = Path(args.wav_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # resource/ created early so everything can write there
    resource_dir = output_dir / "resource"
    resource_dir.mkdir(exist_ok=True)

    # Match *_scored.TextGrid first, fallback to any .TextGrid
    all_tg = sorted(tg_dir.glob("*_scored.TextGrid")) or sorted(tg_dir.glob("*.TextGrid"))
    if not all_tg:
        print(f"No .TextGrid files found in: {tg_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(all_tg)} TextGrid files in {tg_dir}", file=sys.stderr)

    # ── speakers.csv ──
    speakers = load_or_create_speakers_csv(output_dir, all_tg)

    # ── split assignment ──
    if args.test_list:
        with open(args.test_list) as f:
            test_ids = {line.strip() for line in f if line.strip()}
        train_files = [p for p in all_tg if utt_id_from_stem(p.stem) not in test_ids]
        test_files  = [p for p in all_tg if utt_id_from_stem(p.stem) in test_ids]
        print(f"Split: {len(train_files)} train / {len(test_files)} test", file=sys.stderr)
    else:
        train_files = all_tg if args.split == "train" else []
        test_files  = all_tg if args.split == "test"  else []

    # ── resource/scores.json ──
    print("\n── Building scores.json ──", file=sys.stderr)
    scores = {}
    for tg_path in sorted(all_tg):
        utt_id = utt_id_from_stem(tg_path.stem)
        try:
            _, tiers = parse_textgrid(str(tg_path))
            scores[utt_id] = build_scores_entry(tiers)
            print(f"  ✓ {utt_id}", file=sys.stderr)
        except Exception as e:
            print(f"  ✗ {tg_path.name}: {e}", file=sys.stderr)

    with open(resource_dir / "scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(f"  → {len(scores)} entries → resource/scores.json", file=sys.stderr)

    # ── resource/lexicon.txt ──
    print("\n── Building lexicon.txt ──", file=sys.stderr)
    lexicon = build_lexicon(all_tg)
    with open(resource_dir / "lexicon.txt", "w", encoding="utf-8") as f:
        for word in sorted(lexicon):
            f.write(f"{word} {' '.join(lexicon[word])}\n")
    print(f"  → {len(lexicon)} entries → resource/lexicon.txt", file=sys.stderr)

    # ── resource/text-phone ──
    print("\n── Building text-phone ──", file=sys.stderr)
    text_phone_lines = build_text_phone(all_tg)
    with open(resource_dir / "text-phone", "w", encoding="utf-8") as f:
        f.write("\n".join(text_phone_lines) + "\n")
    print(f"  → {len(text_phone_lines)} utterances → resource/text-phone", file=sys.stderr)

    # ── kaldi splits ──
    all_phones: set = set()
    if train_files:
        print("\n── Building train split ──", file=sys.stderr)
        all_phones |= build_split(train_files, wav_dir, output_dir, "train", speakers)
    if test_files:
        print("\n── Building test split ──", file=sys.stderr)
        all_phones |= build_split(test_files, wav_dir, output_dir, "test", speakers)

    print(f"\n✓ Done → {output_dir.resolve()}", file=sys.stderr)
    print("\nOutput structure:", file=sys.stderr)
    print("  dataset/", file=sys.stderr)
    print("  ├── speakers.csv", file=sys.stderr)
    print("  ├── resource/", file=sys.stderr)
    print("  │   ├── lexicon.txt", file=sys.stderr)
    print("  │   ├── phones.txt", file=sys.stderr)
    print("  │   ├── scores.json", file=sys.stderr)
    print("  │   └── text-phone", file=sys.stderr)
    print("  └── data/", file=sys.stderr)
    for split in (["train"] if train_files else []) + (["test"] if test_files else []):
        print(f"      └── {split}/", file=sys.stderr)
        for fname in ["wav.scp", "text", "utt2spk", "spk2utt", "spk2gender", "spk2age"]:
            print(f"          ├── {fname}", file=sys.stderr)


if __name__ == "__main__":
    main()