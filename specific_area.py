#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Dict, Any

# importa utilitários do próprio repo
import utils.file_operations as file_operations
import utils.metrics as metrics

def infer_do_reverse(pairs: List[Dict[str, Any]]) -> bool:
    """Tenta inferir se o run foi em 'duas rodadas' (A,B) e (B,A)."""
    for p in pairs:
        js = p.get("judgments", [])
        if isinstance(js, list) and len(js) >= 2 and js[0] and js[1]:
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Compute metric for a specific area (default: code).")
    parser.add_argument("jsonl", help="Caminho para o arquivo .jsonl gerado pelo JudgeBench (em outputs-br/...).")
    parser.add_argument("--area", default="code", help="Área-alvo (prefix match). Ex.: code, mathematics, knowledge, reasoning.")
    args = parser.parse_args()

    if not os.path.exists(args.jsonl):
        raise SystemExit(f"Arquivo não encontrado: {args.jsonl}")

    # carrega todos os pares do arquivo (histórico completo)
    pairs: List[Dict[str, Any]] = file_operations.read_jsonl(args.jsonl)

    # filtra pela área (startswith, assim como o run_judge.py)
    area = args.area.strip()
    area_pairs = [p for p in pairs if str(p.get("area", "")).startswith(area)]

    if not area_pairs:
        print(f"[{area}] n/a (nenhum par encontrado).")
        return

    # tenta inferir se houve rodada reversa (A,B) e (B,A)
    do_reverse = infer_do_reverse(area_pairs)

    # compute_final_metrics já sabe ler a estrutura dos pares/julgamentos
    try:
        score = metrics.compute_final_metrics(
            area_pairs,
            do_reverse,
            include_fn=lambda _: True  # já filtramos antes
        )
        # também imprime quantos pares entraram no cálculo
        print(f"[{area}] {score:.2f}% (n={len(area_pairs)})")
    except ZeroDivisionError:
        # blindagem extra, caso algo mude em utils.metrics
        print(f"[{area}] n/a (zero pares válidos após filtragem interna).")

if __name__ == "__main__":
    main()
