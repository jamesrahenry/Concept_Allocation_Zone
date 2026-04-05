"""
viz_caz_framework_figures.py — Generate figures for the CAZ Framework paper.

Usage:
    python viz_caz_framework_figures.py

Outputs:
    figures/caz_detection_comparison.png  — Side-by-side single-region vs scored profile
    figures/caz_profile_proof_of_concept.png — Scored CAZ profile for one model×concept

Data requirements:
    Extraction results in ../caz_scaling/results/ for:
    - pythia-1.4b (sentiment, for unimodal example)
    - Qwen2.5-0.5B (credibility, for multimodal example)
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rosetta_tools"))
from rosetta_tools.caz import find_caz_boundary, find_caz_regions_scored, LayerMetrics

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "caz_scaling" / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def classify_caz(peak, score, n_layers):
    """Classify a CAZ by type and assign a color."""
    if peak <= 1:
        return "embedding", "#9467bd"
    elif score >= 0.5:
        return "black hole", "#d62728"
    elif score >= 0.05:
        return "moderate", "#ff7f0e"
    else:
        return "gentle", "#2ca02c"


def load_metrics(caz_json_path):
    """Load CAZ extraction JSON into LayerMetrics list."""
    data = json.load(open(caz_json_path))
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


def find_extraction(model_substr, concept, exclude_instruct=True):
    """Find the extraction directory for a model substring and concept."""
    for d in sorted(RESULTS_ROOT.iterdir()):
        if d.name.startswith(("deepdive", "dark", "manifold")):
            continue
        if exclude_instruct and "Instruct" in d.name:
            continue
        if model_substr in d.name:
            f = d / f"caz_{concept}.json"
            if f.exists():
                return f
    raise FileNotFoundError(f"No extraction found for {model_substr} / {concept}")


# ── Figure 1: Detection comparison ──


def generate_detection_comparison():
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: single-region (unimodal) — pythia-1.4b sentiment
    uni_file = find_extraction("pythia_1.4b", "sentiment")
    data_uni, metrics_uni = load_metrics(uni_file)
    boundary = find_caz_boundary(metrics_uni)
    n_uni = len(metrics_uni)
    seps_uni = [m.separation for m in metrics_uni]
    depths_uni = [100 * m.layer / n_uni for m in metrics_uni]

    ax = axes[0]
    ax.plot(depths_uni, seps_uni, color="#1f77b4", linewidth=2.5)
    ax.fill_between(depths_uni, seps_uni, alpha=0.06, color="#1f77b4")

    start_d = 100 * boundary.caz_start / n_uni
    peak_d = 100 * boundary.caz_peak / n_uni
    end_d = 100 * boundary.caz_end / n_uni
    ax.axvspan(start_d, end_d, alpha=0.2, color="#ff7f0e", label="CAZ region")
    ax.axvline(peak_d, color="#d62728", linestyle="--", linewidth=1.5, label="CAZ peak")
    ax.scatter(
        [peak_d], [seps_uni[boundary.caz_peak]],
        s=120, color="#d62728", zorder=5, edgecolors="white", linewidths=2,
    )

    ax.annotate("Pre-CAZ", (start_d / 2, max(seps_uni) * 0.15),
                ha="center", fontsize=10, color="gray")
    ax.annotate("CAZ", ((start_d + end_d) / 2, max(seps_uni) * 0.15),
                ha="center", fontsize=10, color="#ff7f0e", fontweight="bold")
    ax.annotate("Post-CAZ", ((end_d + 100) / 2, max(seps_uni) * 0.15),
                ha="center", fontsize=10, color="gray")

    model_name = data_uni["model_id"].split("/")[-1]
    ax.set_title(f"Single-Region Detection\nsentiment in {model_name}", fontsize=12)
    ax.set_xlabel("Depth (%)", fontsize=11)
    ax.set_ylabel("Separation S(l)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25)

    # Right: scored profile (multimodal) — Qwen2.5-0.5B credibility
    multi_file = find_extraction("Qwen2.5_0.5B", "credibility")
    data_multi, metrics_multi = load_metrics(multi_file)
    profile = find_caz_regions_scored(metrics_multi)
    n_multi = len(metrics_multi)
    seps_multi = [m.separation for m in metrics_multi]
    depths_multi = [100 * m.layer / n_multi for m in metrics_multi]

    ax = axes[1]
    ax.plot(depths_multi, seps_multi, color="#1f77b4", linewidth=2.5)
    ax.fill_between(depths_multi, seps_multi, alpha=0.06, color="#1f77b4")

    for region in profile.regions:
        cat, color = classify_caz(region.peak, region.caz_score, n_multi)
        start_depth = 100 * region.start / n_multi
        end_depth = 100 * region.end / n_multi
        peak_depth = 100 * region.peak / n_multi
        ax.axvspan(start_depth, end_depth, alpha=0.15, color=color)
        ax.scatter(
            [peak_depth], [seps_multi[region.peak]],
            s=120, color=color, zorder=5, edgecolors="white", linewidths=2,
        )

    model_name2 = data_multi["model_id"].split("/")[-1]
    ax.set_title(f"CAZ Profile (Scored Detection)\ncredibility in {model_name2}", fontsize=12)
    ax.set_xlabel("Depth (%)", fontsize=11)
    ax.set_ylabel("Separation S(l)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)

    legend_elements = [
        Patch(facecolor="#9467bd", alpha=0.4, label="Embedding"),
        Patch(facecolor="#d62728", alpha=0.4, label="Black hole"),
        Patch(facecolor="#ff7f0e", alpha=0.4, label="Moderate"),
        Patch(facecolor="#2ca02c", alpha=0.4, label="Gentle"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    plt.tight_layout()
    out = FIGURES_DIR / "caz_detection_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


# ── Figure 2: Proof-of-concept scored profile ──


def generate_proof_of_concept():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    multi_file = find_extraction("Qwen2.5_0.5B", "credibility")
    data, metrics = load_metrics(multi_file)
    profile = find_caz_regions_scored(metrics)
    n_layers = len(metrics)
    seps = [m.separation for m in metrics]
    depths = [100 * m.layer / n_layers for m in metrics]

    fig, ax = plt.subplots(1, 1, figsize=(12, 5.5))
    ax.plot(depths, seps, color="#1f77b4", linewidth=2.5, zorder=2)
    ax.fill_between(depths, seps, alpha=0.06, color="#1f77b4")

    for region in profile.regions:
        cat, color = classify_caz(region.peak, region.caz_score, n_layers)
        peak_depth = 100 * region.peak / n_layers
        start_depth = 100 * region.start / n_layers
        end_depth = 100 * region.end / n_layers

        ax.axvspan(start_depth, end_depth, alpha=0.15, color=color, zorder=1)
        ax.scatter(
            [peak_depth], [seps[region.peak]],
            s=140, color=color, zorder=5, edgecolors="white", linewidths=2,
        )

        y_off = max(seps) * 0.06
        if seps[region.peak] > max(seps) * 0.7:
            y_off = -max(seps) * 0.08
        ax.annotate(
            f"{cat}\n({region.caz_score:.2f})",
            (peak_depth, seps[region.peak] + y_off),
            ha="center", fontsize=8.5, color=color, fontweight="bold",
        )

    ax.set_xlabel("Depth (%)", fontsize=13)
    ax.set_ylabel("Separation S(l)", fontsize=13)
    ax.set_title(
        f"CAZ Profile: credibility in {data['model_id'].split('/')[-1]}\n"
        f"{len(profile.regions)} CAZes detected — black holes, moderate, gentle, and embedding",
        fontsize=13,
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)

    legend_elements = [
        Line2D([0], [0], color="#1f77b4", linewidth=2.5, label="Separation S(l)"),
        Patch(facecolor="#9467bd", alpha=0.4, label="Embedding CAZ"),
        Patch(facecolor="#d62728", alpha=0.4, label="Black hole (score > 0.5)"),
        Patch(facecolor="#ff7f0e", alpha=0.4, label="Moderate (0.05 - 0.5)"),
        Patch(facecolor="#2ca02c", alpha=0.4, label="Gentle (score < 0.05)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    out = FIGURES_DIR / "caz_profile_proof_of_concept.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    generate_detection_comparison()
    generate_proof_of_concept()
    print("Done.")
