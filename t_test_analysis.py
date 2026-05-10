"""
t-test analysis for JudgeBench-BR mixed-model results.

Computes per-area and overall accuracy for each judge model, groups them
by size (large / medium / small), and runs independent two-sample t-tests
between groups — reproducing the statistical analysis in the paper.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from scipy import stats

# ── Model size classification ───────────────────────────────────────────────
MODEL_SIZE = {
    # Large (> 30 B)
    "deepseek_deepseek-v3.2":        "large",
    "google_gemini-2.5-pro":         "large",
    "moonshotai_kimi-k2-0905":       "large",
    "openai_gpt-5.4":                "large",
    "sabia-3.1":                     "large",
    # Medium (14 – 30 B)
    "meta-llama_llama-4-maverick":   "medium",
    "openai_gpt-oss-20b":            "medium",
    "qwen_qwen3-14b":                "medium",
    "mistralai_ministral-14b-2512":  "medium",
    "mistralai_ministral-8b-2512":   "medium",
    # Small (≤ 13 B)
    "google_gemma-3-12b-it":         "small",
    "meta-llama_llama-3.1-8b-instruct": "small",
    "ibm-granite_granite-4.0-h-micro":  "small",
    # Excluded (safety / guard models — not general judges)
    "meta-llama_llama-guard-4-12b":  "exclude",
}

AREAS = ["Knowledge", "mathematics", "reasoning", "code"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def read_jsonl(path: str) -> List[dict]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    # de-duplicate by pair_id (keep last)
    seen: Dict[str, dict] = {}
    for p in pairs:
        seen[p["pair_id"]] = p
    return list(seen.values())


def flip(decision: Optional[str]) -> Optional[str]:
    if decision == "A>B":
        return "B>A"
    if decision == "B>A":
        return "A>B"
    return decision


def accuracy(pairs: List[dict], area: str = "") -> Optional[float]:
    """Compute accuracy for a subset of pairs (filtered by area prefix)."""
    subset = [p for p in pairs if p.get("area", "").startswith(area)] if area else pairs
    n = len(subset)
    if n == 0:
        return None

    n_correct = 0
    for pair in subset:
        label = pair["label"]
        judgments = pair.get("judgments", [])
        j1 = judgments[0] if len(judgments) > 0 else None
        j2 = judgments[1] if len(judgments) > 1 else None

        d1 = j1["decision"] if j1 else None
        d2 = flip(j2["decision"] if j2 else None)

        counter = 0
        for d in [d1, d2]:
            if d == label:
                counter += 1
            elif d == flip(label):
                counter -= 1

        if counter > 0:
            n_correct += 1

    return 100.0 * n_correct / n


def model_key_from_filename(fname: str) -> Tuple[str, str, str]:
    """
    Parse filename → (judge_name, model_key, display_name).
    Expected pattern: ...,judge_name=<jn>,judge_model=<jm>.jsonl
    """
    stem = os.path.splitext(os.path.basename(fname))[0]

    m_jn = re.search(r"judge_name=([^,]+)", stem)
    m_jm = re.search(r"judge_model=([^,]+)$", stem)

    judge_name = m_jn.group(1) if m_jn else "unknown"
    raw_model  = m_jm.group(1) if m_jm else stem

    # strip "openrouter:" prefix for key lookup
    model_key = raw_model.replace("openrouter:", "")
    display   = model_key.replace("_", "/").replace("openai/", "")
    return judge_name, model_key, display


# ── T-test reporting ──────────────────────────────────────────────────────────

def run_ttest(a: List[float], b: List[float], label_a: str, label_b: str) -> None:
    if len(a) < 2 or len(b) < 2:
        print(f"  {label_a} vs {label_b}: not enough data points ({len(a)} vs {len(b)})")
        return
    t, p = stats.ttest_ind(a, b)
    if p < 0.001:
        p_str = "p < 0.001"
    elif p < 0.01:
        p_str = f"p = {p:.3f}"
    else:
        p_str = f"p = {p:.2f}"
    sig = "✓ significant" if p < 0.05 else "✗ not significant"
    print(f"  {label_a:6s} vs {label_b:6s}:  t = {t:6.2f},  {p_str}  [{sig}]")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    output_dir = "./outputs-br-2"
    files = sorted(
        f for f in os.listdir(output_dir) if f.endswith(".jsonl")
    )

    # results[judge_name][model_key] = {area: acc, "overall": acc}
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for fname in files:
        path = os.path.join(output_dir, fname)
        judge_name, model_key, display = model_key_from_filename(fname)

        size = MODEL_SIZE.get(model_key)
        if size == "exclude":
            continue
        if size is None:
            print(f"[WARN] Unknown model '{model_key}' — skipping.")
            continue

        pairs = read_jsonl(path)
        if not pairs:
            continue

        row: Dict[str, float] = {}
        for area in AREAS:
            acc = accuracy(pairs, area)
            if acc is not None:
                row[area] = acc
        row["overall"] = accuracy(pairs)

        results.setdefault(judge_name, {})[model_key] = row

    # ── Print tables ──────────────────────────────────────────────────────────
    for judge_name in sorted(results):
        print(f"\n{'='*70}")
        print(f"  Prompt: {judge_name.upper()}")
        print(f"{'='*70}")

        header = f"{'Model':<40} {'Know':>7} {'Math':>7} {'Reas':>7} {'Code':>7} {'Overall':>9}  Size"
        print(header)
        print("-" * len(header))

        size_accs: Dict[str, List[float]] = {"large": [], "medium": [], "small": []}

        for model_key in sorted(results[judge_name]):
            row  = results[judge_name][model_key]
            size = MODEL_SIZE.get(model_key, "?")

            know = f"{row['Knowledge']:.2f}%" if "Knowledge" in row else "  N/A  "
            math = f"{row['mathematics']:.2f}%" if "mathematics" in row else "  N/A  "
            reas = f"{row['reasoning']:.2f}%"   if "reasoning"   in row else "  N/A  "
            code = f"{row['code']:.2f}%"        if "code"        in row else "  N/A  "
            ovr  = f"{row['overall']:.2f}%"

            display = model_key.replace("openai_", "").replace("_", "/")
            print(f"  {display:<38} {know:>8} {math:>8} {reas:>8} {code:>8} {ovr:>9}  {size}")

            if size in size_accs:
                size_accs[size].append(row["overall"])

        # ── T-tests ───────────────────────────────────────────────────────────
        print(f"\n  --- T-tests (Overall accuracy, independent two-sample) ---")
        for size, accs in size_accs.items():
            print(f"  {size.capitalize():6s}: {[round(a,2) for a in accs]}")
        print()
        run_ttest(size_accs["large"],  size_accs["medium"], "large",  "medium")
        run_ttest(size_accs["large"],  size_accs["small"],  "large",  "small")
        run_ttest(size_accs["medium"], size_accs["small"],  "medium", "small")

    print()


if __name__ == "__main__":
    main()
