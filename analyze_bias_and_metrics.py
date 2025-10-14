import json
import argparse
from typing import List, Dict, Any

try:
    from sklearn.metrics import classification_report
except ImportError:
    print("Scikit-learn não foi encontrada. Por favor, instale-a com: pip install scikit-learn")
    exit()

def flip_judgment(decision: str) -> str:
    """Inverte uma decisão de 'A>B' para 'B>A' e vice-versa."""
    if not decision:
        return None
    if decision == "A>B":
        return "B>A"
    elif decision == "B>A":
        return "A>B"
    return decision

def analyze_results(pairs: List[Dict[Any, Any]], reverse_order: bool = True):
    """
    Calcula e imprime métricas detalhadas a partir de uma lista de pares julgados.
    """
    total_pairs_in_category = len(pairs)
    print("="*50)
    print(f"Total pairs in file: {total_pairs_in_category}")
    print("="*50)

    if total_pairs_in_category == 0:
        print("No pairs found to analyze.")
        return

    # A lógica aqui é focada no modo de avaliação dupla (reverse_order=True)
    if not reverse_order:
        print("This script is designed for dual-judgment (reverse_order=True) files.")
        return

    n_correct, n_incorrect, n_tie = 0, 0, 0
    n_nulls, n_inconsistent = 0, 0
    n_first_choice_bias = 0
    y_true, y_pred = [], []

    for pair in pairs:
        label = pair.get("label")
        judgments = pair.get("judgments", [None, None])
        if len(judgments) < 2: judgments.extend([None] * (2 - len(judgments)))
        judgment1, judgment2 = judgments

        raw_decision1 = judgment1.get("decision") if judgment1 else None
        raw_decision2 = judgment2.get("decision") if judgment2 else None
        
        decision1 = raw_decision1
        decision2 = flip_judgment(raw_decision2)

        if decision1 is None or decision2 is None:
            n_nulls += 1
        else:
            if decision1 != decision2:
                n_inconsistent += 1
            
            if raw_decision1 == "A>B" and raw_decision2 == "A>B":
                n_first_choice_bias += 1
            
            if decision1 in ["A>B", "B>A"]:
                y_true.append(label)
                y_pred.append(decision1)
            if decision2 in ["A>B", "B>A"]:
                y_true.append(label)
                y_pred.append(decision2)

        counter = 0
        for decision in [decision1, decision2]:
            if decision == label: counter += 1
            elif decision == flip_judgment(label): counter -= 1
        if counter > 0: n_correct += 1
        elif counter < 0: n_incorrect += 1
        else: n_tie += 1

    pairs_with_valid_response = total_pairs_in_category - n_nulls
    
    # --- IMPRESSÃO DO RELATÓRIO ---
    print("\n--- Overall Performance Summary ---")
    print(f"  - Pairs with valid model response: {pairs_with_valid_response} of {total_pairs_in_category}")
    print(f"  - General inconsistency (A!=B): {n_inconsistent} of {pairs_with_valid_response} ({ (n_inconsistent/pairs_with_valid_response*100) if pairs_with_valid_response > 0 else 0 :.1f}%)")
    print(f"  - Strong First-Position Bias: {n_first_choice_bias} of {pairs_with_valid_response} ({ (n_first_choice_bias/pairs_with_valid_response*100) if pairs_with_valid_response > 0 else 0 :.1f}%)")
    print(f"  - Final Accuracy Score: {100 * n_correct / total_pairs_in_category if total_pairs_in_category > 0 else 0:.2f}%")
    print(f"  - Accounted pairs (Correct / Incorrect / Ties): {n_correct} / {n_incorrect} / {n_tie}")
    
    print("\n" + "-"*15 + " Classification Metrics (per judgment) " + "-"*15)
    if y_true:
        print(classification_report(y_true, y_pred, labels=["A>B", "B>A"], zero_division=0))
    else:
        print("  No valid judgments to generate a classification report.")
    print("-" * 52 + "\n")

def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Lê um arquivo JSONL e retorna uma lista de dicionários."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
        exit()
    except json.JSONDecodeError as e:
        print(f"Erro ao decodificar JSON no arquivo '{file_path}': {e}")
        exit()
    return data

if __name__ == "__main__":
    # Configura o parser de argumentos da linha de comando
    parser = argparse.ArgumentParser(description="Calculate advanced metrics for JudgeBench results.")
    parser.add_argument("jsonl_file", type=str, help="Path to the .jsonl file with judging results.")
    args = parser.parse_args()

    # Lê e processa o arquivo de resultados
    all_pairs = read_jsonl(args.jsonl_file)

    # Lógica de desduplicação para garantir que apenas o último julgamento de cada par seja usado
    if all_pairs:
        print(f"De-duplicating results... Found {len(all_pairs)} total entries.")
        unique_pairs_dict = {pair['pair_id']: pair for pair in all_pairs}
        unique_pairs = list(unique_pairs_dict.values())
        print(f"Processing {len(unique_pairs)} unique pairs for metric calculation.\n")
        analyze_results(unique_pairs)
    else:
        print("The provided file is empty.")