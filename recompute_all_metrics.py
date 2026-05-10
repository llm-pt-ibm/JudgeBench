#!/usr/bin/env python3
import argparse
import os
from typing import List, Dict, Any

import utils.file_operations as file_operations
import utils.metrics as metrics

AREAS = ["code", "mathematics", "knowledge", "reasoning"]

def infer_do_reverse(pairs: List[Dict[str, Any]]) -> bool:
    """Detecta rapidamente se houve rodada reversa (A,B) e (B,A)."""
    for p in pairs:
        js = p.get("judgments", [])
        if isinstance(js, list) and len(js) >= 2 and js[0] and js[1]:
            return True
    return False

def compute_subset(label: str, pairs: List[Dict[str, Any]], do_reverse: bool) -> None:
    subset = [p for p in pairs if str(p.get("area", "")).startswith(label)] if label else pairs
    if not subset:
        print(f"{label or 'Overall'}: n/a (no pairs)")
        return
    try:
        score = metrics.compute_final_metrics(subset, do_reverse, include_fn=lambda _: True)
        print(f"{label or 'Overall'}: {score:.2f}% (n={len(subset)})")
    except ZeroDivisionError:
        print(f"{label or 'Overall'}: n/a (zero valid pairs)")

def main():
    parser = argparse.ArgumentParser(description="Recompute metrics for all areas and overall.")
    parser.add_argument("jsonl", help="Caminho para o arquivo .jsonl gerado (ex.: outputs-br/xxx.jsonl)")
    args = parser.parse_args()

    if not os.path.exists(args.jsonl):
        raise SystemExit(f"Arquivo não encontrado: {args.jsonl}")

    pairs: List[Dict[str, Any]] = file_operations.read_jsonl(args.jsonl)
    if not pairs:
        print("Arquivo sem pares. Nada a computar.")
        return

    do_reverse = infer_do_reverse(pairs)

    for area in AREAS:
        compute_subset(area, pairs, do_reverse)

    compute_subset("", pairs, do_reverse)  # Overall

if __name__ == "__main__":
    main()
