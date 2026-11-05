"""Deterministic summaries/plots from retained observations; no synthetic data."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ("pbrs", "shifted", "smooth_pbrs", "smooth_shifted")
LABELS = {"unshaped": "Unshaped", "pbrs": "PBRS", "shifted": "Shifted",
          "smooth_pbrs": "Smoothed PBRS", "smooth_shifted": "Smoothed shifted"}


def load_runs(root):
    result = []
    for phase in ("reconstruction", "core"):
        for path in sorted((Path(root)/phase).glob("*.json")):
            if path.name == "manifest.json": continue
            rec = json.loads(path.read_text())
            if "run_id" not in rec: continue
            rec["_path"] = str(path.with_suffix(".npz"))
            result.append(rec)
    return result


def interval(values, seed=1818):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    resampled = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    low, high = np.quantile(resampled, [.025, .975])
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)),
            "ci_low": float(low), "ci_high": float(high), "n_seed_blocks": len(values),
            "seed_block_values": values.tolist()}


def core_key(c):
    return (c["map_name"], c["normalisation"], c["corruption"], c["q_initial"], c["seed"], c["method"])


def contrast(core, method_a, method_b, normalisation=None, corruption=None):
    table = {core_key(r["config"]): r["metrics"]["mean_checkpoint_return"] for r in core}
    blocks = defaultdict(list)
    for key, value in table.items():
        map_name, norm, frac, q0, seed, method = key
        if method != method_a or (normalisation is not None and norm != normalisation) or (corruption is not None and frac != corruption):
            continue
        other = (map_name, norm, frac, q0, seed, method_b)
        blocks[seed].append(value-table[other])
    return interval([np.mean(blocks[s]) for s in sorted(blocks)])


def write_csv(path, rows):
    if not rows: raise ValueError("cannot write empty scientific summary")
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def analyse(root="results", destination="analysis"):
    root, destination = Path(root), Path(destination)
    destination.mkdir(exist_ok=True)
    figures = destination/"figures"; figures.mkdir(exist_ok=True)
    runs = load_runs(root)
    core = [r for r in runs if r["config"]["phase"] == "core"]
    reconstruction = [r for r in runs if r["config"]["phase"] == "reconstruction"]
    expected = {p:json.loads((root/p/"manifest.json").read_text())["runs"] for p in ("core", "reconstruction")}
    assert len(core) == expected["core"] and len(reconstruction) == expected["reconstruction"], "incomplete results"
    rows = []
    for r in runs:
        rows.append({**r["config"], **r["metrics"], "run_id": r["run_id"],
                     "total_seconds": r["timing"]["total_seconds_including_save"]})
    write_csv(destination/"run_summary.csv", rows)
    summary = {"counts": {"core_runs": len(core), "reconstruction_runs": len(reconstruction),
                           "training_steps": sum(r["metrics"]["training_steps"] for r in runs),
                           "evaluation_steps": sum(r["metrics"]["evaluation_steps"] for r in runs),
                           "evaluation_episodes": sum(r["metrics"]["evaluation_episodes"] for r in runs),
                           "summed_run_seconds": sum(r["timing"]["total_seconds_including_save"] for r in runs)},
               "extension_primary": contrast(core, "smooth_shifted", "shifted"),
               "shift_contrast": contrast(core, "shifted", "pbrs"),
               "smoothing_unshifted": contrast(core, "smooth_pbrs", "pbrs"),
               "by_normalisation": {}, "by_corruption": {}}
    for norm in ("literal", "unit"):
        summary["by_normalisation"][norm] = contrast(core, "smooth_shifted", "shifted", normalisation=norm)
    for frac in (0., .25, .75):
        summary["by_corruption"][str(frac)] = contrast(core, "smooth_shifted", "shifted", corruption=frac)

    # Panels retain Q initialisation and normalisation instead of hiding them.
    colours = {"pbrs":"#6d6d6d", "shifted":"#1769aa", "smooth_pbrs":"#c47a17", "smooth_shifted":"#26734d"}
    fig, axes = plt.subplots(2, 3, figsize=(8.0, 4.3), sharex=True)
    core_table = {core_key(r["config"]):r for r in core}
    for row, norm in enumerate(("literal", "unit")):
        for col, q0 in enumerate((-1., 0., 1.)):
            ax = axes[row, col]
            for method in METHODS:
                means, sds = [], []
                for frac in (0., .25, .75):
                    values = []
                    for seed in range(10):
                        values.append(np.mean([core_table[(m,norm,frac,q0,seed,method)]["metrics"]["mean_checkpoint_return"]
                                               for m in ("open10", "detour10")]))
                    means.append(np.mean(values)); sds.append(np.std(values, ddof=1))
                ax.errorbar([0,.25,.75], means, yerr=sds, label=LABELS[method], color=colours[method],
                            marker="o", markersize=3, linewidth=1.1, capsize=2)
            unshaped = [np.mean([core_table[(m,"unit",0.,q0,seed,"unshaped")]["metrics"]["mean_checkpoint_return"]
                                 for m in ("open10", "detour10")]) for seed in range(10)]
            ax.axhline(np.mean(unshaped), color="#202020", linestyle="--", linewidth=.9, label="Unshaped")
            ax.set_title(f"{norm}; initial Q={q0:g}", fontsize=9)
            ax.grid(alpha=.2); ax.tick_params(labelsize=8)
            if col==0: ax.set_ylabel("Mean checkpoint return", fontsize=8)
            if row==1: ax.set_xlabel("Selected corruption fraction", fontsize=8)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=7.5, bbox_to_anchor=(.5,1.02))
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(figures/"core_comparison.pdf", bbox_inches="tight")
    fig.savefig(figures/"core_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Full factor-level summaries and failures remain available outside the report.
    grouped = defaultdict(list)
    for r in reconstruction:
        c=r["config"]
        grouped[(c["reward"],c["cutoff"],c["normalisation"],c["q_initial"],c["method"])].append(r)
    source_rows=[]
    for key, group in sorted(grouped.items()):
        source_rows.append({**dict(zip(("reward","cutoff","normalisation","q_initial","method"),key)),
            "runs":len(group), "mcr_mean":float(np.mean([r["metrics"]["mean_checkpoint_return"] for r in group])),
            "mcr_sd":float(np.std([r["metrics"]["mean_checkpoint_return"] for r in group],ddof=1)),
            "final_success_mean":float(np.mean([r["metrics"]["final_success"] for r in group])),
            "zero_final_success_runs":sum(r["metrics"]["final_success"]==0 for r in group)})
    write_csv(destination/"reconstruction_factor_summary.csv", source_rows)
    summary["reconstruction_factors"] = source_rows
    reliability=[]
    for r in core:
        c=r["config"]
        if c["method"] not in ("pbrs","smooth_pbrs") or c["q_initial"] != 0: continue
        reliability.append({"map_name":c["map_name"],"normalisation":c["normalisation"],
                            "corruption":c["corruption"],"seed":c["seed"],"method":c["method"],
                            **r["potential"]["after"]})
    write_csv(destination/"potential_diagnostics.csv", reliability)
    summary["core_failure_counts"] = {method:sum(r["metrics"]["final_success"]==0 for r in core if r["config"]["method"]==method)
                                       for method in ("unshaped",*METHODS)}
    (destination/"summary.json").write_text(json.dumps(summary, indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("reconstruction_factors",)}, indent=2))
    return summary


if __name__ == "__main__": analyse()
