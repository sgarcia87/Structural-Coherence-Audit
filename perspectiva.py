#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal demo (v2): Structural Coherence Loss under Perspective Shift
Adds:
- Stability checks (ratio + balance)
- Dominance (hub collapse)
- Avg balance + unstable perspectives report

C(G) = 1 - mean pairwise JSD(pi_phi_i || pi_phi_j),
where pi_phi is normalized combined field (pi_A + pi_B)/2.

Equilibrium per perspective: top1 by score = (pa+pb) - λ|pa-pb|
Stability: balance <= balance_max AND ratio(top1/top2) >= ratio_min
"""

import math
import networkx as nx
from itertools import combinations
from collections import Counter

# --------------------------
# Utilities
# --------------------------

def ppr(G: nx.DiGraph, source: str, alpha: float = 0.85) -> dict:
    pers = {n: 0.0 for n in G.nodes()}
    pers[source] = 1.0
    return nx.pagerank(G, alpha=alpha, personalization=pers, weight="weight")

def normalize_dist(d: dict, nodes: list[str]) -> list[float]:
    vec = [float(d.get(n, 0.0)) for n in nodes]
    s = sum(vec)
    if s <= 0:
        return [1.0 / len(nodes)] * len(nodes)
    return [v / s for v in vec]

def kl(p, q, eps=1e-12):
    out = 0.0
    for pi, qi in zip(p, q):
        pi = max(eps, float(pi))
        qi = max(eps, float(qi))
        out += pi * math.log(pi / qi)
    return out

def jsd(p, q):
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def equilibrium_score(pa: float, pb: float, lam: float = 0.6) -> float:
    return (pa + pb) - lam * abs(pa - pb)

def stability(top1, top2, ratio_min=1.35, balance_max=0.80):
    """
    top1/top2: tuples (node, score, pa, pb)
    """
    if top1 is None:
        return False, {"balance": 1.0, "ratio": None}

    _, s1, pa1, pb1 = top1
    bal = abs(pa1 - pb1) / max(1e-12, (pa1 + pb1))

    ratio = None
    if top2 is not None:
        _, s2, _, _ = top2
        ratio = (s1 / s2) if s2 > 0 else float("inf")

    ok = True
    if ratio is not None and ratio < ratio_min:
        ok = False
    if bal > balance_max:
        ok = False

    return ok, {"balance": bal, "ratio": ratio}

def find_equilibrium(G: nx.DiGraph, A: str, B: str, *, alpha=0.85, lam=0.6):
    nodes = list(G.nodes())
    pa = ppr(G, A, alpha=alpha)
    pb = ppr(G, B, alpha=alpha)

    scored = []
    for n in nodes:
        if n in (A, B):
            continue
        s = equilibrium_score(pa.get(n, 0.0), pb.get(n, 0.0), lam=lam)
        scored.append((n, float(s), float(pa.get(n, 0.0)), float(pb.get(n, 0.0))))

    scored.sort(key=lambda x: x[1], reverse=True)
    top1 = scored[0] if scored else None
    top2 = scored[1] if len(scored) > 1 else None

    return top1, top2, pa, pb

def combined_field_distribution(nodes, pa, pb):
    comb = {n: 0.5 * float(pa.get(n, 0.0)) + 0.5 * float(pb.get(n, 0.0)) for n in nodes}
    return normalize_dist(comb, nodes)

# --------------------------
# Graph construction
# --------------------------

def build_G0() -> nx.DiGraph:
    G = nx.DiGraph()
    nodes = ["realidad","modelo","teoria","observacion","medicion","prediccion","dato","interpretacion"]
    G.add_nodes_from(nodes)

    edges = [
        ("observacion","realidad"),
        ("medicion","observacion"),
        ("dato","medicion"),
        ("modelo","dato"),
        ("prediccion","modelo"),
        ("teoria","modelo"),
        ("interpretacion","teoria"),
        ("interpretacion","dato"),
    ]
    for u,v in edges:
        G.add_edge(u, v, weight=1.0, tipo="estructura")
    return G

def contaminate_G1(G0: nx.DiGraph) -> nx.DiGraph:
    G = G0.copy()
    contaminating_edges = [
        ("prediccion","realidad"),
        ("teoria","medicion"),
        ("interpretacion","realidad"),
    ]
    for u,v in contaminating_edges:
        if not G.has_edge(u, v):
            G.add_edge(u, v, weight=1.0, tipo="estructura_contaminante")
    return G

# --------------------------
# Experiment
# --------------------------

def run_experiment(G: nx.DiGraph, perspectives, *, alpha=0.85, lam=0.6, ratio_min=1.35, balance_max=0.80):
    nodes = list(G.nodes())

    per = []      # (A,B,top1,stable,stats)
    pi_phis = []  # distributions for C(G)

    balances = []
    stable_flags = []

    for (A,B) in perspectives:
        top1, top2, pa, pb = find_equilibrium(G, A, B, alpha=alpha, lam=lam)
        ok, stats = stability(top1, top2, ratio_min=ratio_min, balance_max=balance_max)

        per.append((A, B, top1, ok, stats))
        pi_phi = combined_field_distribution(nodes, pa, pb)
        pi_phis.append(pi_phi)

        balances.append(stats["balance"])
        stable_flags.append(ok)

    # Fragmentation (unique equilibria / m)
    eq_nodes = [x[2][0] for x in per if x[2] is not None]
    frag = len(set(eq_nodes)) / max(1, len(eq_nodes))

    # Dominance: max frequency of the same equilibrium node
    counts = Counter(eq_nodes)
    dom = (max(counts.values()) / len(eq_nodes)) if eq_nodes else 0.0
    dom_node = counts.most_common(1)[0][0] if counts else None

    # Coherence C(G) via mean pairwise JSD between pi_phi distributions
    if len(pi_phis) < 2:
        mean_jsd = 0.0
    else:
        jsds = []
        for i, j in combinations(range(len(pi_phis)), 2):
            jsds.append(jsd(pi_phis[i], pi_phis[j]))
        mean_jsd = sum(jsds) / len(jsds)

    C = 1.0 - mean_jsd

    stable_rate = sum(1 for x in stable_flags if x) / len(stable_flags) if stable_flags else 0.0
    avg_balance = sum(balances) / len(balances) if balances else 1.0

    return {
        "per": per,
        "frag": frag,
        "dom": dom,
        "dom_node": dom_node,
        "stable_rate": stable_rate,
        "avg_balance": avg_balance,
        "mean_jsd": mean_jsd,
        "C": C,
    }

def pretty_print(label, result):
    print(f"\n=== {label} ===")
    print(f"Coherence C(G):      {result['C']:.4f}   (mean JSD={result['mean_jsd']:.4f})")
    print(f"Fragmentation (Frag): {result['frag']:.3f}")
    print(f"Dominance (Dom):      {result['dom']:.3f}  (node={result['dom_node']})")
    print(f"Stable rate:          {result['stable_rate']:.3f}")
    print(f"Avg balance:          {result['avg_balance']:.3f}")

    print("\nPer-perspective equilibria:")
    for (A,B,top1,ok,stats) in result["per"]:
        n, s, pa, pb = top1
        ratio = stats["ratio"]
        ratio_s = "n/a" if ratio is None else f"{ratio:.2f}"
        print(f" - ({A:12s},{B:14s}) -> eq={n:14s} score={s:.6f} bal={stats['balance']:.2f} ratio={ratio_s}  {'OK' if ok else 'UNSTABLE'}")

    unstable = [(A,B,stats) for (A,B,_,ok,stats) in result["per"] if not ok]
    if unstable:
        print("\n⚠️ Unstable perspectives:")
        for (A,B,stats) in unstable:
            ratio = stats["ratio"]
            ratio_s = "n/a" if ratio is None else f"{ratio:.2f}"
            print(f"   - ({A},{B}): bal={stats['balance']:.2f} ratio={ratio_s}")

def main():
    perspectives = [
        ("dato","teoria"),
        ("observacion","prediccion"),
        ("medicion","interpretacion"),
    ]

    alpha = 0.85
    lam = 0.6
    ratio_min = 1.35
    balance_max = 0.80

    G0 = build_G0()
    G1 = contaminate_G1(G0)

    r0 = run_experiment(G0, perspectives, alpha=alpha, lam=lam, ratio_min=ratio_min, balance_max=balance_max)
    r1 = run_experiment(G1, perspectives, alpha=alpha, lam=lam, ratio_min=ratio_min, balance_max=balance_max)

    pretty_print("G0 (coherent)", r0)
    pretty_print("G1 (contaminated)", r1)

    print("\n=== Summary ===")
    print(f"C(G0)={r0['C']:.4f} vs C(G1)={r1['C']:.4f}")
    print(f"StableRate(G0)={r0['stable_rate']:.3f} vs StableRate(G1)={r1['stable_rate']:.3f}")
    print(f"Dom(G0)={r0['dom']:.3f} vs Dom(G1)={r1['dom']:.3f}  (dom-node G1={r1['dom_node']})")

    if r1["C"] < r0["C"]:
        print("✅ Coherence drops after minimal contamination.")
    else:
        print("⚠️ Coherence did not drop; adjust contamination edges or perspectives.")

    if r1["dom"] > r0["dom"]:
        print("✅ Dominance increases → indicates hub-collapse / degenerate convergence.")
    else:
        print("ℹ️ Dominance did not increase.")

if __name__ == "__main__":
    main()
