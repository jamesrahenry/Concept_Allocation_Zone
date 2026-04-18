"""
viz_caz_framework_figures.py — Generate figures for the CAZ Framework paper.

Usage:
    python viz_caz_framework_figures.py

Outputs:
    figures/caz_detection_comparison.png  — Side-by-side single-region vs scored profile
    figures/caz_profile_proof_of_concept.png — Scored CAZ profile for one model×concept

Style: conforms to Rosetta_Program/VIZ_STYLE_GUIDE.md via caz_scaling/src/viz_style.py.
Written: 2026-04-05  Updated: 2026-04-18 UTC
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rosetta_tools"))
sys.path.insert(0, str(ROOT / "caz_scaling" / "src"))

from rosetta_tools.caz import find_caz_boundary, find_caz_regions_scored, LayerMetrics
from viz_style import (
    concept_color,
    CAZ_CAT_COLORS, CAZ_CAT_FILL, CAZ_CAT_LABELS, caz_score_cat,
    THEME, apply_theme, layer_ticks, add_outside_callouts,
)

RESULTS_ROOT = ROOT / "caz_scaling" / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Embedding CAZ (layer ≤ 1) is orthogonal to score categories; deep purple.
EMBED_COLOR = "#7B1FA2"
EMBED_FILL  = "#E1BEE7"
EMBED_LABEL = "Embedding"

SEP_LW = 2.2


def caz_viz_style(peak: int, score: float) -> tuple[str, str, str, str]:
    """Return (key, edge_color, fill_color, label) for a CAZ region.

    Embedding (peak ≤ 1) overrides the score-category classification.
    """
    if peak <= 1:
        return "embedding", EMBED_COLOR, EMBED_FILL, EMBED_LABEL
    cat = caz_score_cat(score)
    return cat, CAZ_CAT_COLORS[cat], CAZ_CAT_FILL[cat], CAZ_CAT_LABELS[cat]


def load_metrics(path: Path):
    data = json.load(open(path))
    metrics = [
        LayerMetrics(
            layer=m["layer"],
            separation=m["separation_fisher"],
            coherence=m["coherence"],
            velocity=m.get("velocity", 0.0),
        )
        for m in data["layer_data"]["metrics"]
    ]
    return data, metrics


def find_extraction(model_substr: str, concept: str) -> Path:
    for d in sorted(RESULTS_ROOT.iterdir()):
        if d.name.startswith(("deepdive", "dark", "manifold")):
            continue
        if "Instruct" in d.name:
            continue
        if model_substr in d.name:
            f = d / f"caz_{concept}.json"
            if f.exists():
                return f
    raise FileNotFoundError(f"No extraction for {model_substr} / {concept}")


def draw_profile(ax, metrics, concept: str, *, marker_size: int = 100, score_text: bool = True):
    """Plot separation line, score text inside markers. Returns list of region dicts."""
    seps = [m.separation for m in metrics]
    layers = [m.layer for m in metrics]
    n = len(metrics)
    color = concept_color(concept)

    ax.plot(layers, seps, color=color, linewidth=SEP_LW, zorder=3)
    ax.fill_between(layers, seps, alpha=0.07, color=color, zorder=1)

    positions, labels = layer_ticks(n)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(0, n - 1)
    ax.set_ylim(bottom=0)
    apply_theme(ax)

    return seps, layers, n


def mark_region(ax, region, seps, *, marker_size: int = 100, score_text: bool = True):
    key, edge, fill, label = caz_viz_style(region.peak, region.caz_score)
    ax.axvspan(region.start, region.end, alpha=0.35, color=fill, zorder=1)
    ax.scatter(
        [region.peak], [seps[region.peak]],
        s=marker_size, facecolor=edge, edgecolor="white", linewidths=1.4,
        zorder=6,
    )
    if score_text:
        ax.annotate(
            f"{region.caz_score:.2f}",
            xy=(region.peak, seps[region.peak]),
            ha="center", va="center",
            fontsize=5.5, fontweight="bold", color="white",
            zorder=7,
        )
    return key, edge, label


# ─── Figure 1: Detection comparison ────────────────────────────────────────────

def _panel_title(ax, text):
    ax.text(
        0.015, 0.97, text,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=10.5, fontweight="bold", color=THEME["text"],
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
        zorder=8,
    )


def generate_detection_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    # Left: single-region (velocity boundary) — pythia-1.4b sentiment
    uni_file = find_extraction("pythia_1.4b", "sentiment")
    data_uni, metrics_uni = load_metrics(uni_file)
    boundary = find_caz_boundary(metrics_uni)

    ax = axes[0]
    seps_u, _, n_u = draw_profile(ax, metrics_uni, "sentiment")

    # Add headroom so top callouts don't collide with the in-axes title
    ax.set_ylim(0, max(seps_u) * 1.22)

    region_color = "#37474F"
    ax.axvspan(boundary.caz_start, boundary.caz_end,
               alpha=0.18, color=region_color, zorder=1)
    ax.scatter(
        [boundary.caz_peak], [seps_u[boundary.caz_peak]],
        s=110, facecolor=region_color, edgecolor="white", linewidths=1.5,
        zorder=6,
    )

    _panel_title(ax, "(a) Single-Region Detection — sentiment in Pythia-1.4B")
    ax.set_ylabel("Separation $S(\\ell)$", fontsize=10)
    ax.set_xlabel("Layer (depth %)", fontsize=10)

    # Inline CAZ peak label next to the dot
    ax.annotate(
        "CAZ peak",
        xy=(boundary.caz_peak, seps_u[boundary.caz_peak]),
        xytext=(8, -4), textcoords="offset points",
        ha="left", va="top",
        fontsize=8, fontweight="bold", color=region_color,
        zorder=7,
    )
    add_outside_callouts(ax, [
        {"x": (boundary.caz_start + boundary.caz_end) / 2,
         "y": seps_u[boundary.caz_start],
         "label": "CAZ region\n(velocity threshold)", "color": region_color},
    ], n_layers=n_u)

    # Right: scored profile — Qwen2.5-0.5B credibility
    multi_file = find_extraction("Qwen2.5_0.5B", "credibility")
    data_multi, metrics_multi = load_metrics(multi_file)
    profile = find_caz_regions_scored(metrics_multi)

    ax = axes[1]
    seps_m, _, n_m = draw_profile(ax, metrics_multi, "credibility")
    ax.set_ylim(0, max(seps_m) * 1.18)

    for region in profile.regions:
        mark_region(ax, region, seps_m, marker_size=110)

    _panel_title(ax, "(b) Scored Profile — credibility in Qwen2.5-0.5B")
    ax.set_ylabel("Separation $S(\\ell)$", fontsize=10)
    ax.set_xlabel("Layer (depth %)", fontsize=10)

    # Figure-level legend: 5 patches (embedding + 4 score categories)
    legend_elems = [
        Patch(facecolor=EMBED_FILL,                edgecolor=EMBED_COLOR,
              linewidth=1.0, label=EMBED_LABEL),
        Patch(facecolor=CAZ_CAT_FILL["black_hole"], edgecolor=CAZ_CAT_COLORS["black_hole"],
              linewidth=1.0, label="Black hole (>0.5)"),
        Patch(facecolor=CAZ_CAT_FILL["strong"],     edgecolor=CAZ_CAT_COLORS["strong"],
              linewidth=1.0, label="Strong (>0.2)"),
        Patch(facecolor=CAZ_CAT_FILL["moderate"],   edgecolor=CAZ_CAT_COLORS["moderate"],
              linewidth=1.0, label="Moderate (>0.05)"),
        Patch(facecolor=CAZ_CAT_FILL["gentle"],     edgecolor=CAZ_CAT_COLORS["gentle"],
              linewidth=1.0, label="Gentle (≤0.05)"),
    ]
    fig.legend(
        handles=legend_elems, loc="upper center",
        bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False, fontsize=8.5,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIGURES_DIR / "caz_detection_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")
    plt.close(fig)


# ─── Figure 2: Proof-of-concept — Qwen credibility enlarged ────────────────────

def generate_proof_of_concept():
    multi_file = find_extraction("Qwen2.5_0.5B", "credibility")
    data, metrics = load_metrics(multi_file)
    profile = find_caz_regions_scored(metrics)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5.2))
    seps, _, n = draw_profile(ax, metrics, "credibility")

    callouts = []
    for region in profile.regions:
        key, edge, label = mark_region(ax, region, seps, marker_size=130)
        # Callouts only for the embedding and black hole, to avoid clutter.
        if key == "embedding":
            callouts.append({
                "x": region.peak, "y": seps[region.peak],
                "label": f"Embedding CAZ\n(layer {region.peak}, {region.caz_score:.2f})",
                "color": edge,
            })
        elif key == "black_hole":
            callouts.append({
                "x": region.peak, "y": seps[region.peak],
                "label": f"Black hole\n(layer {region.peak}, {region.caz_score:.2f})",
                "color": edge,
            })
        elif key == "strong":
            callouts.append({
                "x": region.peak, "y": seps[region.peak],
                "label": f"Strong\n(layer {region.peak}, {region.caz_score:.2f})",
                "color": edge,
            })

    if callouts:
        add_outside_callouts(ax, callouts, n_layers=n)

    ax.set_title(
        f"Scored CAZ Profile — credibility in Qwen2.5-0.5B "
        f"({len(profile.regions)} CAZes detected)",
        fontsize=11, loc="left", pad=8,
    )
    ax.set_ylabel("Separation $S(\\ell)$", fontsize=10)
    ax.set_xlabel("Layer (depth %)", fontsize=10)

    legend_elems = [
        Line2D([0], [0], color=concept_color("credibility"), linewidth=SEP_LW,
               label="Separation $S(\\ell)$"),
        Patch(facecolor=EMBED_FILL, edgecolor=EMBED_COLOR, linewidth=1.0,
              label="Embedding"),
        Patch(facecolor=CAZ_CAT_FILL["black_hole"], edgecolor=CAZ_CAT_COLORS["black_hole"],
              linewidth=1.0, label="Black hole (>0.5)"),
        Patch(facecolor=CAZ_CAT_FILL["strong"], edgecolor=CAZ_CAT_COLORS["strong"],
              linewidth=1.0, label="Strong (>0.2)"),
        Patch(facecolor=CAZ_CAT_FILL["moderate"], edgecolor=CAZ_CAT_COLORS["moderate"],
              linewidth=1.0, label="Moderate (>0.05)"),
        Patch(facecolor=CAZ_CAT_FILL["gentle"], edgecolor=CAZ_CAT_COLORS["gentle"],
              linewidth=1.0, label="Gentle (≤0.05)"),
    ]
    fig.legend(
        handles=legend_elems, loc="upper center",
        bbox_to_anchor=(0.5, 1.02), ncol=6, frameon=False, fontsize=8.5,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIGURES_DIR / "caz_profile_proof_of_concept.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    generate_detection_comparison()
    generate_proof_of_concept()
    print("Done.")
