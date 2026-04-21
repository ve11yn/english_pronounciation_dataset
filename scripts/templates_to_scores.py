"""
textgrid_to_scores.py
=====================
Converts annotated _scoring.TextGrid files → speechocean762 scores.json

Tiers expected (from inject_scoring_tiers.py output):
    1. sentence_acc
    2. sentence_completeness
    3. sentence_fluency
    4. sentence_prosody
    5. words              (pre-filled from MFA)
    6. word_acc
    7. word_stress
    8. phones             (pre-filled from MFA)
    9. phone_acc

Calculated (not in TextGrid):
    - word_total   = average(word_acc, word_stress)
    - sentence_total = average(sentence_acc, sentence_completeness,
                               sentence_fluency, sentence_prosody)

Usage:
    python textgrid_to_scores.py --input 00001_scoring.TextGrid
    python textgrid_to_scores.py --input_dir ./annotated --output scores.json
"""

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_textgrid(path: str):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    duration_match = re.search(r'xmax\s*=\s*([0-9.]+)', content)
    duration = float(duration_match.group(1)) if duration_match else 0.0

    tiers = {}
    tier_blocks = re.split(r'\s+item\s*\[\d+\]:', content)

    for block in tier_blocks[1:]:
        name_match = re.search(r'name\s*=\s*"([^"]*)"', block)
        if not name_match:
            continue
        tier_name = name_match.group(1).strip()
        intervals = []
        for ib in re.split(r'\s+intervals\s*\[\d+\]:', block)[1:]:
            xmin = float(re.search(r'xmin\s*=\s*([0-9.e+-]+)', ib).group(1))
            xmax = float(re.search(r'xmax\s*=\s*([0-9.e+-]+)', ib).group(1))
            text_match = re.search(r'text\s*=\s*"([^"]*)"', ib)
            text = text_match.group(1).strip() if text_match else ""
            intervals.append({"xmin": xmin, "xmax": xmax, "text": text})
        tiers[tier_name] = intervals

    return duration, tiers

SILENCE = {"", "sp", "sil", "SIL", "<eps>", "spn"}

def scored(intervals):
    """Non-silence intervals only."""
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
    val = s[0]["text"]
    return safe_int(val, tier_name, 0) if as_int else safe_float(val, tier_name, 0)

def check_required(tiers, path):
    required = [
        "sentence_acc", "sentence_completeness", "sentence_fluency", "sentence_prosody",
        "words", "word_acc", "word_stress", "phones", "phone_acc",
    ]
    missing = [t for t in required if t not in tiers]
    if missing:
        raise ValueError(f"Missing tiers in {path}: {missing}")


def textgrid_to_entry(path: str, utt_id: str = None) -> dict:
    _, tiers = parse_textgrid(path)
    check_required(tiers, path)

    # --- Sentence scores ---
    acc          = single_score(tiers["sentence_acc"],          "sentence_acc")
    completeness = single_score(tiers["sentence_completeness"], "sentence_completeness", as_int=False)
    fluency      = single_score(tiers["sentence_fluency"],      "sentence_fluency")
    prosodic     = single_score(tiers["sentence_prosody"],      "sentence_prosody")
    total        = round((acc + completeness + fluency + prosodic) / 4, 2)

    # --- Word intervals ---
    word_ivs      = scored(tiers["words"])
    word_acc_ivs  = scored(tiers["word_acc"])
    word_str_ivs  = scored(tiers["word_stress"])

    n = len(word_ivs)
    for name, ivs in [("word_acc", word_acc_ivs), ("word_stress", word_str_ivs)]:
        if len(ivs) != n:
            raise ValueError(f"Tier '{name}' has {len(ivs)} intervals but 'words' has {n}")

    # --- Phone alignment ---
    phone_ivs     = scored(tiers["phones"])
    phone_acc_ivs = scored(tiers["phone_acc"])

    if len(phone_ivs) != len(phone_acc_ivs):
        raise ValueError(
            f"'phones' has {len(phone_ivs)} intervals but 'phone_acc' has {len(phone_acc_ivs)}"
        )

    phone_data = [
        {
            "xmin":  p["xmin"],
            "xmax":  p["xmax"],
            "label": p["text"],
            "acc":   safe_float(a["text"], "phone_acc", i),
        }
        for i, (p, a) in enumerate(zip(phone_ivs, phone_acc_ivs))
    ]

    # --- Build words ---
    sentence_text = " ".join(iv["text"].upper() for iv in word_ivs)
    words_out = []

    for i, wiv in enumerate(word_ivs):
        w_phones = [
            p for p in phone_data
            if p["xmin"] >= wiv["xmin"] - 1e-6
            and p["xmax"] <= wiv["xmax"] + 1e-6
        ]
        w_acc    = safe_int(word_acc_ivs[i]["text"],  "word_acc",    i)
        w_stress = safe_int(word_str_ivs[i]["text"],  "word_stress", i)
        w_total  = round((w_acc + w_stress) / 2, 2)

        words_out.append({
            "text":            wiv["text"].upper(),
            "accuracy":        w_acc,
            "stress":          w_stress,
            "total":           w_total,
            "phones":          [p["label"] for p in w_phones],
            "phones-accuracy": [p["acc"]   for p in w_phones],
            "mispronunciations": [],
        })

    uid = utt_id or Path(path).stem.replace("_scoring", "")

    return {
        "accuracy":     acc,
        "completeness": completeness,
        "fluency":      fluency,
        "prosodic":     prosodic,
        "text":         sentence_text,
        "total":        total,
        "words":        words_out,
    }, uid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert annotated _scoring.TextGrid files to scores.json"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",     help="Single .TextGrid file")
    group.add_argument("--input_dir", help="Directory of .TextGrid files")
    parser.add_argument("--output",   help="Output scores.json (omit to print)")
    args = parser.parse_args()

    files = [args.input] if args.input else sorted(Path(args.input_dir).glob("*.TextGrid"))
    if not files:
        print(f"No .TextGrid files found.", file=sys.stderr)
        sys.exit(1)

    entries = {}
    for f in files:
        try:
            entry, uid = textgrid_to_entry(str(f))
            entries[uid] = entry
            print(f"✓ {f}", file=sys.stderr)
        except Exception as e:
            print(f"✗ {f}: {e}", file=sys.stderr)

    result = json.dumps(entries, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as out:
            out.write(result)
        print(f"\nSaved {len(entries)} entries → {args.output}", file=sys.stderr)
    else:
        print(result)


if __name__ == "__main__":
    main()