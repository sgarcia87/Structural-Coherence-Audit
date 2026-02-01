#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 1 — Silent Structural Coherence Degradation
Visualizes:
A) G0 with distributed equilibria
B) G1 with contaminating edges
C) G1 with degenerate hub convergence
"""

import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

# --------------------------
# Graph construction (same as experiment)
# --------------------------

def build_G0():
    G = nx.DiGraph()
    nodes = ["realidad","modelo","teoria","observacion",
             "medicion","prediccion","dato","interpretacion"]
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
        G.add_edge(u,v, tipo="estructura")
    return G

def contaminate_G1(G0):
    G = G0.copy()
    contaminated = [
        ("prediccion","realidad"),
        ("teoria","medicion"),
        ("interpretacion","realidad"),
    ]
    for u,v in contaminated:
        G.add_edge(u,v, tipo="contaminante")
    return G, contaminated

# --------------------------
# Drawing utilities
# --------------------------

def draw_graph(ax, G, pos, eq_counts=None, highlight_edges=None, title=""):
    eq_counts = eq_counts or {}
    maxc = max(eq_counts.values()) if eq_counts else 1

    node_sizes = []
    node_colors = []

    for n in G.nodes():
        c = eq_counts.get(n, 0)
        node_sizes.append(800 + 2000 * (c / maxc))
        node_colors.append("#ff6666" if c > 0 else "#dddddd")

    # Draw base graph
    nx.draw(
        G, pos,
        ax=ax,
        with_labels=True,
        node_size=node_sizes,
        node_color=node_colors,
        edge_color="#999999",
        arrows=True,
        arrowsize=15,
        font_size=10
    )

    # Highlight contaminating edges
    if highlight_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=highlight_edges,
            ax=ax,
            edge_color="red",
            width=3,
            arrows=True,
            arrowsize=18
        )

    ax.set_title(title, fontsize=12)
    ax.axis("off")

# --------------------------
# Main figure
# --------------------------

def main():
    # Equilibria from your experiment (B configuration)
    # G0 equilibria: medicion, realidad, observacion
    eq_G0 = Counter({
        "medicion": 1,
        "realidad": 1,
        "observacion": 1,
    })

    # G1 equilibria: realidad dominates (2 of 3)
    eq_G1 = Counter({
        "realidad": 2,
        "medicion": 1,
    })

    G0 = build_G0()
    G1, contaminated_edges = contaminate_G1(G0)

    # Fixed layout for comparability
    pos = nx.spring_layout(G0, seed=42)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A
    draw_graph(
        axes[0], G0, pos,
        eq_counts=eq_G0,
        title="(A) Coherent graph G₀\nDistributed equilibrium structure"
    )

    # Panel B
    draw_graph(
        axes[1], G1, pos,
        eq_counts=None,
        highlight_edges=contaminated_edges,
        title="(B) Minimally contaminated graph G₁\nStructural perturbations (red)"
    )

    # Panel C
    draw_graph(
        axes[2], G1, pos,
        eq_counts=eq_G1,
        title="(C) G₁ under perspective shift\nDegenerate hub convergence"
    )

    plt.tight_layout()
    plt.savefig("figure1_structural_coherence.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
