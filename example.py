#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
perspectiva_v2.py

Paper companion (minimal reference implementation)
==================================================

This script reproduces the core experiment from the paper:

- G0 (coherent baseline) vs G1 (contaminated) in a single run (default).
- Perspective-induced dynamics via Personalized PageRank (PPR).
- Structural coherence via mean pairwise Jensen–Shannon divergence (JSD).
- Equilibrium detection via balanced interference score.

Design goals:
- Minimal degrees of freedom (paper-grade reproducibility).
- No auto-policy heuristics (those belong to v3 tooling).
- Optional Figure-1-like visualization.

Typical usage
-------------
# 1) Reproduce paper experiment (G0 vs G1)
python3 perspectiva_v2.py

# 2) Same + Figure 1 for both graphs
python3 perspectiva_v2.py --figure1

# 3) Run on a real JSON graph (single-graph mode)
python3 perspectiva_v2.py --json graph.json --auto_perspectives 6
python3 perspectiva_v2.py --json graph.json --persp "calor,frío" --persp "presión,volumen" --figure1

JSON format
-----------
Node-link style:
- nodes: [{"id": ...}, ...]
- links: [{"source": ..., "target": ..., "weight": ...}, ...]
Also supports NetworkX-style 'edges' instead of 'links'.
"""

import argparse
import math
import json
import os
from itertools import combinations
from collections import Counter

import networkx as nx

import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt


# --------------------------
# Core math
# --------------------------

def ppr(G: nx.DiGraph, source: str, alpha: float = 0.85) -> dict:
    pers = {n: 0.0 for n in G.nodes()}
    pers[source] = 1.0
    return nx.pagerank(G, alpha=alpha, personalization=pers, weight="weight")


def normalize_dist(d: dict, nodes: list[str]) -> list[float]:
    vec = [float(d.get(n, 0.0)) for n in nodes]
    s = sum(vec)
    if s <= 0:
        # Minimal fallback for degenerate cases (rare in connected graphs).
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
# Virtual graphs (paper baseline)
# --------------------------

def build_G0() -> nx.DiGraph:
    G = nx.DiGraph()
    nodes = ["realidad", "modelo", "teoria", "observacion", "medicion", "prediccion", "dato", "interpretacion"]
    G.add_nodes_from(nodes)

    edges = [
        ("observacion", "realidad"),
        ("medicion", "observacion"),
        ("dato", "medicion"),
        ("modelo", "dato"),
        ("prediccion", "modelo"),
        ("teoria", "modelo"),
        ("interpretacion", "teoria"),
        ("interpretacion", "dato"),
    ]
    for u, v in edges:
        G.add_edge(u, v, weight=1.0, tipo="estructura")
    return G


def contaminate_G1(G0: nx.DiGraph):
    G = G0.copy()
    contaminating_edges = [
        ("prediccion", "realidad"),
        ("teoria", "medicion"),
        ("interpretacion", "realidad"),
    ]
    for u, v in contaminating_edges:
        if not G.has_edge(u, v):
            G.add_edge(u, v, weight=1.0, tipo="estructura_contaminante", color="red")
    return G, contaminating_edges


# --------------------------
# JSON IO (single-graph mode)
# --------------------------

def load_graph_json(path: str) -> nx.DiGraph:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes") or []
    links = data.get("links") or data.get("edges") or []

    G = nx.DiGraph()

    for nd in nodes:
        nid = nd.get("id") if isinstance(nd, dict) else nd
        if nid is None:
            continue
        attrs = dict(nd) if isinstance(nd, dict) else {}
        attrs.pop("id", None)
        G.add_node(str(nid), **attrs)

    for e in links:
        if not isinstance(e, dict):
            continue
        u = e.get("source")
        v = e.get("target")
        if u is None or v is None:
            continue

        attrs = dict(e)
        w = attrs.get("weight", attrs.get("base_weight", 1.0))
        try:
            w = float(w)
        except Exception:
            w = 1.0
        attrs["weight"] = w

        G.add_edge(str(u), str(v), **attrs)

    if G.number_of_nodes() == 0:
        raise ValueError(f"JSON has no valid nodes: {path}")

    return G


def save_graph_json(G: nx.DiGraph, path: str):
    nodes = []
    for n, attrs in G.nodes(data=True):
        item = {"id": n}
        item.update(attrs or {})
        nodes.append(item)

    links = []
    for u, v, attrs in G.edges(data=True):
        item = {"source": u, "target": v}
        item.update(attrs or {})
        links.append(item)

    data = {"directed": True, "multigraph": False, "graph": {}, "nodes": nodes, "links": links}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# --------------------------
# Perspectives (minimal)
# --------------------------

def pick_auto_perspectives_degree(G: nx.DiGraph, k: int = 3) -> list[tuple[str, str]]:
    """Minimal degree-based auto-pick (no policy heuristics; paper companion)."""
    if k <= 0:
        return []
    deg = [(n, G.in_degree(n) + G.out_degree(n)) for n in G.nodes()]
    deg.sort(key=lambda x: x[1], reverse=True)
    cand = [x[0] for x in deg[: max(6, k * 4)]]
    pairs = []
    for a, b in combinations(cand, 2):
        pairs.append((a, b))
        if len(pairs) >= k:
            break
    if len(pairs) < k:
        nodes = list(G.nodes())
        for a, b in combinations(nodes, 2):
            if (a, b) not in pairs:
                pairs.append((a, b))
            if len(pairs) >= k:
                break
    return pairs[:k]


# --------------------------
# Experiment + reporting
# --------------------------

def run_experiment(G: nx.DiGraph, perspectives, *, alpha=0.85, lam=0.6, ratio_min=1.35, balance_max=0.80):
    nodes = list(G.nodes())

    per = []      # (A,B,top1,stable,stats)
    pi_phis = []  # distributions for C(G)
    balances = []
    stable_flags = []

    for (A, B) in perspectives:
        if A not in G or B not in G:
            continue

        top1, top2, pa, pb = find_equilibrium(G, A, B, alpha=alpha, lam=lam)
        ok, stats = stability(top1, top2, ratio_min=ratio_min, balance_max=balance_max)

        per.append((A, B, top1, ok, stats))
        pi_phi = combined_field_distribution(nodes, pa, pb)
        pi_phis.append(pi_phi)

        balances.append(stats["balance"])
        stable_flags.append(ok)

    eq_nodes = [x[2][0] for x in per if x[2] is not None]
    frag = len(set(eq_nodes)) / max(1, len(eq_nodes))

    counts = Counter(eq_nodes)
    dom = (max(counts.values()) / len(eq_nodes)) if eq_nodes else 0.0
    dom_node = counts.most_common(1)[0][0] if counts else None

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
        "eq_counts": Counter(eq_nodes),
    }


def pretty_print(label, result):
    print(f"\n=== {label} ===")
    print(f"Coherence C(G):       {result['C']:.4f}   (mean JSD={result['mean_jsd']:.4f})")
    print(f"Fragmentation (Frag): {result['frag']:.3f}")
    print(f"Dominance (Dom):      {result['dom']:.3f}  (node={result['dom_node']})")
    print(f"Stable rate:          {result['stable_rate']:.3f}")
    print(f"Avg balance:          {result['avg_balance']:.3f}")

    print("\nPer-perspective equilibria:")
    for (A, B, top1, ok, stats) in result["per"]:
        if top1 is None:
            print(f" - ({A},{B}) -> eq=None")
            continue
        n, s, pa, pb = top1
        ratio = stats["ratio"]
        ratio_s = "n/a" if ratio is None else f"{ratio:.2f}"
        print(f" - ({A:16s},{B:16s}) -> eq={n:18s} score={s:.6f} bal={stats['balance']:.2f} ratio={ratio_s}  {'OK' if ok else 'UNSTABLE'}")


# --------------------------
# Figure 1 (optional)
# --------------------------

def _edge_is_contaminating(attrs: dict) -> bool:
    t = str(attrs.get("tipo", "")).lower()
    c = str(attrs.get("color", "")).lower()
    return ("contamin" in t) or (c == "red")


def draw_graph(ax, G, pos, eq_counts=None, highlight_edges=None, title=""):
    eq_counts = eq_counts or Counter()
    maxc = max(eq_counts.values()) if eq_counts else 1

    node_sizes = []
    node_colors = []
    for n in G.nodes():
        c = eq_counts.get(n, 0)
        node_sizes.append(300 + 1700 * (c / maxc) if maxc > 0 else 300)
        node_colors.append("#ff6666" if c > 0 else "#dddddd")

    nx.draw(
        G, pos,
        ax=ax,
        with_labels=True,
        node_size=node_sizes,
        node_color=node_colors,
        edge_color="#999999",
        arrows=True,
        arrowsize=12,
        font_size=9
    )

    if highlight_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=highlight_edges,
            ax=ax,
            edge_color="red",
            width=2.5,
            arrows=True,
            arrowsize=14
        )

    ax.set_title(title, fontsize=11)
    ax.axis("off")


def run_figure1_like(G_base: nx.DiGraph, *, contaminated_edges=None, eq_counts=None, out_png="figure1_structural_coherence.png"):
    pos = nx.spring_layout(G_base, seed=42)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    draw_graph(axes[0], G_base, pos, eq_counts=None, highlight_edges=None, title="(A) Graph\nBase structure")

    if contaminated_edges is None:
        contaminated_edges = [(u, v) for u, v, a in G_base.edges(data=True) if _edge_is_contaminating(a)]
    draw_graph(axes[1], G_base, pos, eq_counts=None, highlight_edges=contaminated_edges, title="(B) Graph\nContaminating edges (red)")

    draw_graph(axes[2], G_base, pos, eq_counts=eq_counts or Counter(), highlight_edges=None, title="(C) Graph under perspective shift\nEquilibrium convergence (node size)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)
    return out_png


# --------------------------
# CLI
# --------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Paper companion experiment: structural coherence under perspective shift (PPR + JSD)."
    )

    ap.add_argument("--json", default=None, help="Path to a node-link JSON graph (nodes+links or nodes+edges).")
    ap.add_argument("--save_virtual", default=None, help="Save the default virtual graphs (G0/G1) to JSON with this base path.")

    ap.add_argument("--persp", action="append", default=None, metavar="A,B",
                    help="Manual perspective pair 'A,B'. Repeatable. Example: --persp 'dato,teoria'")
    ap.add_argument("--auto_perspectives", type=int, default=0, metavar="K",
                    help="If K>0, auto-pick K perspectives by node degree (minimal heuristic).")

    ap.add_argument("--alpha", type=float, default=0.85, help="PageRank damping factor.")
    ap.add_argument("--lam", type=float, default=0.6, help="Balance penalty λ used in the equilibrium score.")
    ap.add_argument("--ratio_min", type=float, default=1.35, help="Stability threshold: top1/top2 >= ratio_min.")
    ap.add_argument("--balance_max", type=float, default=0.80, help="Stability threshold: |pa-pb|/(pa+pb) <= balance_max.")

    ap.add_argument("--figure1", action="store_true", help="Generate Figure-1-like PNGs.")
    ap.add_argument("--out_png", default="figure1_structural_coherence.png",
                    help="Base output name for --figure1 (will append _G0/_G1 in paper mode).")

    return ap.parse_args()


def main():
    args = parse_args()

    # Build perspectives helper
    def build_perspectives(G: nx.DiGraph):
        perspectives = []
        if args.persp:
            for s in args.persp:
                if not s or "," not in s:
                    continue
                a, b = [x.strip() for x in s.split(",", 1)]
                if a and b:
                    perspectives.append((a, b))
        if args.auto_perspectives and args.auto_perspectives > 0:
            perspectives.extend(pick_auto_perspectives_degree(G, args.auto_perspectives))
        if not perspectives:
            # Paper default perspectives for virtual graphs
            demo = [("dato", "teoria"), ("observacion", "prediccion"), ("medicion", "interpretacion")]
            if all(a in G and b in G for a, b in demo):
                perspectives = demo
            else:
                perspectives = pick_auto_perspectives_degree(G, 3)
        return perspectives

    # -------------------------------------------------------
    # Mode A (default): Paper reproduction (G0 vs G1)
    # -------------------------------------------------------
    if not args.json:
        G0 = build_G0()
        G1, contaminated = contaminate_G1(G0)

        if args.save_virtual:
            base = args.save_virtual
            if base.lower().endswith(".json"):
                base = base[:-5]
            save_graph_json(G0, f"{base}_G0.json")
            save_graph_json(G1, f"{base}_G1.json")
            print(f"💾 Saved virtual graphs to: {base}_G0.json and {base}_G1.json")

        print(f"Virtual G0  (n={G0.number_of_nodes()} e={G0.number_of_edges()})")
        print(f"Virtual G1  (n={G1.number_of_nodes()} e={G1.number_of_edges()})")

        perspectives = build_perspectives(G0)
        print(f"🧭 Perspectives ({len(perspectives)}): {perspectives}")

        r0 = run_experiment(G0, perspectives, alpha=args.alpha, lam=args.lam, ratio_min=args.ratio_min, balance_max=args.balance_max)
        r1 = run_experiment(G1, perspectives, alpha=args.alpha, lam=args.lam, ratio_min=args.ratio_min, balance_max=args.balance_max)

        pretty_print("G0 (baseline)", r0)
        pretty_print("G1 (contaminated)", r1)

        # Deltas
        print("\n=== Summary (G1 - G0) ===")
        print(f"Δ Coherence C(G):   {r1['C'] - r0['C']:+.4f}")
        print(f"Δ Fragmentation:    {r1['frag'] - r0['frag']:+.3f}")
        print(f"Δ Dominance:        {r1['dom'] - r0['dom']:+.3f}")
        print(f"Δ Stable rate:      {r1['stable_rate'] - r0['stable_rate']:+.3f}")
        print(f"Δ Avg balance:      {r1['avg_balance'] - r0['avg_balance']:+.3f}")

        if args.figure1:
            stem, ext = os.path.splitext(args.out_png)
            ext = ext or ".png"
            out0 = f"{stem}_G0{ext}"
            out1 = f"{stem}_G1{ext}"
            run_figure1_like(G0, contaminated_edges=[], eq_counts=r0["eq_counts"], out_png=out0)
            run_figure1_like(G1, contaminated_edges=contaminated, eq_counts=r1["eq_counts"], out_png=out1)
            print(f"🖼️  Saved: {out0}")
            print(f"🖼️  Saved: {out1}")

        return

    # -------------------------------------------------------
    # Mode B: Single graph (JSON)
    # -------------------------------------------------------
    G = load_graph_json(args.json)
    print(f"Loaded graph: {args.json}  (n={G.number_of_nodes()} e={G.number_of_edges()})")

    perspectives = build_perspectives(G)
    print(f"🧭 Perspectives ({len(perspectives)}): {perspectives}")

    r = run_experiment(G, perspectives, alpha=args.alpha, lam=args.lam, ratio_min=args.ratio_min, balance_max=args.balance_max)
    pretty_print("Result", r)

    if args.figure1:
        out = run_figure1_like(G, contaminated_edges=None, eq_counts=r["eq_counts"], out_png=args.out_png)
        print(f"🖼️  Saved: {out}")


if __name__ == "__main__":
    main()
