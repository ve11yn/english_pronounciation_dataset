"""
Tiers produced (in order):
    1.  sentence_acc          — 0-10  (9-10 excellent, 7-8 good, 5-6 ok, 3-4 poor, 0-2 fail)
    2.  sentence_completeness — 0-10  (proportion of words attempted)
    3.  sentence_fluency      — 0-10  (8-10 fluent, 6-7 minor pauses, 4-5 many pauses, 0-3 fail)
    4.  sentence_prosody      — 0-10  (9-10 correct intonation, 7-8 nearly correct, 3-6 unstable, 0-2 fail)
    5.  words                 — pre-filled from MFA (UPPERCASE)
    6.  word_acc              — 0-10  (10 correct, 7-9 heavy accent, 4-6 <30% wrong, 2-3 >30% wrong, 0-1 fail)
    7.  word_stress           — 0 or 10  (10 correct/monosyllable, 5 wrong stress)
    8.  phones                — pre-filled from MFA (ARPAbet)
    9.  phone_acc             — 0-2   (2 correct, 1 heavy accent, 0 wrong/missed)

Usage:
    python textgrid_to_template.py --input 00001.TextGrid
    python textgrid_to_template.py --input 00001.TextGrid --output 00001_scoring.TextGrid
    python textgrid_to_template.py --input_dir ./mfa_output --output_dir ./to_annotate
"""

import argparse
import re
import sys
from pathlib import Path


def parse_textgrid(path: str) -> dict:
    """Parse MFA TextGrid → {tier_name: [(xmin, xmax, text), ...]}"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    header_match = re.search(r'xmax\s*=\s*([0-9.]+)', content)
    duration = float(header_match.group(1)) if header_match else 0.0

    tiers = {}
    tier_blocks = re.split(r'\s+item\s*\[\d+\]:', content)

    for block in tier_blocks[1:]:
        name_match = re.search(r'name\s*=\s*"([^"]*)"', block)
        if not name_match:
            continue
        tier_name = name_match.group(1).strip()
        intervals = []
        interval_blocks = re.split(r'\s+intervals\s*\[\d+\]:', block)
        for ib in interval_blocks[1:]:
            xmin = float(re.search(r'xmin\s*=\s*([0-9.e+-]+)', ib).group(1))
            xmax = float(re.search(r'xmax\s*=\s*([0-9.e+-]+)', ib).group(1))
            text_match = re.search(r'text\s*=\s*"([^"]*)"', ib)
            text = text_match.group(1).strip() if text_match else ""
            intervals.append((xmin, xmax, text))
        tiers[tier_name] = intervals

    return duration, tiers


def format_intervals(intervals: list) -> str:
    lines = [f"        intervals: size = {len(intervals)}"]
    for i, (xmin, xmax, text) in enumerate(intervals, 1):
        lines.append(f"        intervals [{i}]:")
        lines.append(f"            xmin = {xmin}")
        lines.append(f"            xmax = {xmax}")
        lines.append(f'            text = "{text}"')
    return "\n".join(lines)


def write_textgrid(duration: float, ordered_tiers: list, output_path: str):
    """
    ordered_tiers: list of (tier_name, [(xmin, xmax, text), ...])
    """
    lines = []
    lines.append('File type = "ooTextFile"')
    lines.append('Object class = "TextGrid"')
    lines.append("")
    lines.append("xmin = 0")
    lines.append(f"xmax = {duration}")
    lines.append("tiers? <exists>")
    lines.append(f"size = {len(ordered_tiers)}")
    lines.append("item []:")

    for idx, (tier_name, intervals) in enumerate(ordered_tiers, 1):
        lines.append(f"    item [{idx}]:")
        lines.append('        class = "IntervalTier"')
        lines.append(f'        name = "{tier_name}"')
        lines.append("        xmin = 0")
        lines.append(f"        xmax = {duration}")
        lines.append(format_intervals(intervals))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def get_word_intervals(tiers: dict) -> list:
    """Get word intervals from MFA — try 'words' then 'word'."""
    for key in ("words", "word"):
        if key in tiers:
            return tiers[key]
    raise ValueError("No 'words' or 'word' tier found in MFA output.")


def get_phone_intervals(tiers: dict) -> list:
    """Get phone intervals from MFA — try 'phones' then 'phone'."""
    for key in ("phones", "phone"):
        if key in tiers:
            return tiers[key]
    raise ValueError("No 'phones' or 'phone' tier found in MFA output.")


def make_sentence_tier(duration: float, placeholder: str = "") -> list:
    """Single scored interval covering the whole utterance."""
    return [(0.0, duration, placeholder)]


def make_score_tier_like(reference_intervals: list, placeholder: str = "") -> list:
    """
    Mirror the time structure of a reference tier (words or phones)
    but replace all non-empty text with placeholder (blank for annotator).
    Silence intervals stay empty.
    """
    result = []
    for xmin, xmax, text in reference_intervals:
        if text.strip() == "" or text.strip() == "sp" or text.strip() == "SIL":
            result.append((xmin, xmax, ""))
        else:
            result.append((xmin, xmax, placeholder))
    return result


def inject_tiers(input_path: str, output_path: str):
    duration, mfa_tiers = parse_textgrid(input_path)

    word_ivs  = get_word_intervals(mfa_tiers)
    phone_ivs = get_phone_intervals(mfa_tiers)

    # Normalize MFA silence labels → ""
    def clean(ivs):
        cleaned = []
        for xmin, xmax, text in ivs:
            t = text.strip()
            if t in ("", "sp", "SIL", "<eps>", "sil", "spn"):
                cleaned.append((xmin, xmax, ""))
            else:
                cleaned.append((xmin, xmax, t.upper()))
        return cleaned

    word_ivs  = clean(word_ivs)
    phone_ivs = clean(phone_ivs)

    # Sentence-level tiers — annotator fills one number each
    # sentence_total is excluded — calculated later from scores
    sentence_tiers = [
        ("sentence_acc",          make_sentence_tier(duration)),
        ("sentence_completeness", make_sentence_tier(duration)),
        ("sentence_fluency",      make_sentence_tier(duration)),
        ("sentence_prosody",      make_sentence_tier(duration)),
    ]

    # Word tier — pre-filled from MFA
    word_tier = ("words", word_ivs)

    # Word score tiers — same time structure as words, blanks for annotator
    # word_total is excluded — calculated later from scores
    word_score_tiers = [
        ("word_acc",    make_score_tier_like(word_ivs)),
        ("word_stress", make_score_tier_like(word_ivs)),
    ]

    # Phone tier — pre-filled from MFA
    phone_tier = ("phones", phone_ivs)

    # Phone accuracy tier — same time structure as phones, blanks for annotator
    phone_acc_tier = ("phone_acc", make_score_tier_like(phone_ivs))

    ordered_tiers = (
        sentence_tiers
        + [word_tier]
        + word_score_tiers
        + [phone_tier, phone_acc_tier]
    )

    write_textgrid(duration, ordered_tiers, output_path)
    print(f"✓ {input_path} → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inject speechocean762 scoring tiers into MFA TextGrids."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",     help="Single MFA .TextGrid file")
    group.add_argument("--input_dir", help="Directory of MFA .TextGrid files")

    parser.add_argument("--output",     help="Output path (single file mode)")
    parser.add_argument("--output_dir", help="Output directory (batch mode)")
    parser.add_argument(
        "--suffix", default="_scoring",
        help="Suffix added to filename in batch mode (default: _scoring)"
    )
    args = parser.parse_args()

    if args.input:
        out = args.output or str(Path(args.input).with_suffix("")) + "_scoring.TextGrid"
        inject_tiers(args.input, out)

    else:
        in_dir  = Path(args.input_dir)
        out_dir = Path(args.output_dir) if args.output_dir else in_dir / "scoring"
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(in_dir.glob("*.TextGrid"))
        if not files:
            print(f"No .TextGrid files in {in_dir}", file=sys.stderr)
            sys.exit(1)

        ok, fail = 0, 0
        for f in files:
            out = out_dir / (f.stem + args.suffix + ".TextGrid")
            try:
                inject_tiers(str(f), str(out))
                ok += 1
            except Exception as e:
                print(f"✗ {f}: {e}", file=sys.stderr)
                fail += 1

        print(f"\nDone: {ok} OK, {fail} failed → {out_dir}")


if __name__ == "__main__":
    main()