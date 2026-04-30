import time
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from itertools import combinations
import config


def solve(min_support, min_confidence, verbose=True):
    start_time = time.perf_counter()

    # --- KROK 1: Szybkie wczytywanie i filtrowanie danych ---
    try:
        # Wczytujemy InvoiceNo jako string, aby móc sprawdzić isdigit()
        df_raw = pd.read_csv(config.datapath, encoding='ISO-8859-1',
                             usecols=[0, 1], dtype={0: str})

        # Filtrowanie identyczne z slow_solutions:
        # 1. InvoiceNo musi składać się tylko z cyfr
        # 2. StockCode nie może być pusty
        df_clean = df_raw[df_raw.iloc[:, 0].str.isdigit() & df_raw.iloc[:, 1].notna()]

        # Grupowanie unikalnych produktów na fakturę[cite: 1, 2]
        transactions = df_clean.groupby(df_clean.columns[0])[df_clean.columns[1]].unique().tolist()
    except Exception as e:
        if verbose: print(f"Błąd wczytywania: {e}")
        return []

    # --- KROK 2: One-Hot Encoding ---
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_bin = pd.DataFrame(te_ary, columns=te.columns_)

    # --- KROK 3: Generowanie zbiorów częstych ---
    # fpgrowth zapewnia optymalną wydajność
    frequent_itemsets = fpgrowth(df_bin, min_support=min_support, use_colnames=True)

    # Przekształcenie w słownik dla dostępu O(1) podczas generowania reguł
    support_map = {frozenset(item): supp for item, supp in
                   zip(frequent_itemsets['itemsets'], frequent_itemsets['support'])}

    # --- KROK 4: Ręczne generowanie reguł (ZGODNIE Z ZADANIEM) ---
    rules = []
    # Iterujemy po wszystkich zbiorach o rozmiarze >= 2
    for itemset, support in support_map.items():
        if len(itemset) < 2:
            continue

        items = list(itemset)
        # Ręczna generacja podzbiorów dla lewej strony reguły (A)
        for r in range(1, len(items)):
            for a_tuple in combinations(items, r):
                antecedent = frozenset(a_tuple)
                consequent = itemset - antecedent

                # Obliczanie ufności na podstawie mapy wsparcia
                support_a = support_map.get(antecedent)

                if support_a:
                    confidence = support / support_a
                    if confidence >= min_confidence:
                        rules.append({
                            'A': list(antecedent),
                            'B': list(consequent),
                            'supp': float(support),
                            'conf': float(confidence)
                        })

    execution_time = time.perf_counter() - start_time

    # --- KROK 5: Raportowanie i wypisywanie wszystkich reguł ---
    if verbose:
        print("\n" + "=" * 50)
        print(f"RAPORT: FAST SOLUTION")
        print(f"Czas wykonania: {execution_time:.4f} sekund")
        print(f"Liczba reguł: {len(rules)}")
        print("-" * 50)
        for rule in rules:
            ant = "{" + ", ".join(rule['A']) + "}"
            cons = "{" + ", ".join(rule['B']) + "}"
            print(f"{ant:30} => {cons:30} | supp: {rule['supp']:.3f} | conf: {rule['conf']:.3f}")
        print("=" * 50 + "\n")

    return rules

if __name__ == "__main__":
    solve(config.min_support, config.min_confidence, verbose=True)