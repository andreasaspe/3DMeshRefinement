"""
inspect_metrics.py
------------------
Inspect refinement delta metrics from metrics_summary.csv.

Usage:
    python inspect_metrics.py                        # prompts for file
    python inspect_metrics.py path/to/metrics.csv   # direct path

Produces:
    1. Console summary table (before / after / delta for every metric)
    2. Delta bar chart (green = improved, red = worse)
    3. Before vs after grouped bar chart
    4. Normalised improvement boxplot
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["savefig.dpi"] = 150
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["font.size"] = 10

# ── Config ─────────────────────────────────────────────────────────────
METRICS = ["Dice", "EAT_Dice", "NSD", "HD", "HD95", "ASD", "ASSD", "volume_overlap_inside", "volume_overlap_outside"]


HIGHER_IS_BETTER = {"Dice": True, "EAT_Dice": True,
                    "ASD": False, "ASSD": False,
                    "HD": False, "HD95": False, "NSD": True,
                    "volume_overlap_inside": False, "volume_overlap_outside": False}


# ── Load CSV ────────────────────────────────────────────────────────────
csv_path = "/data/awias/periseg/saros/TS_pericardium/pytorch3d/metrics/best_grid_search_result_EXCLUDEGRID_EAT0/metrics_summary_taubin.csv"

if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"File not found: {csv_path}")

df = pd.read_csv(csv_path)
out_dir = os.path.dirname(os.path.abspath(csv_path))
print(f"\nLoaded {len(df)} series from {csv_path}\n")

# ── Convert Dice/EAT_Dice/NSD to % immediately after loading ───────────
for m in ["Dice", "EAT_Dice", "NSD"]:
    for prefix in ["before_", "after_"]:
        col = f"{prefix}{m}"
        if col in df.columns:
            df[col] *= 100

# Convert inside and outside mm3 to cm3 for better readability
for prefix in ["before_", "after_"]:
    for suffix in ["volume_overlap_inside", "volume_overlap_outside"]:
        col = f"{prefix}{suffix}"
        if col in df.columns:
            df[col] /= 1000  # convert mm³ to cm³

# Short x-axis labels (last two underscore-segments or full name)
short_labels = ["_".join(s.split("_")[-2:]) if len(s) > 25 else s for s in df["series"]]
x = np.arange(len(df))


# ── 1. Console table ────────────────────────────────────────────────────
SEP = "=" * 90
print(SEP)
print(f"{'Series':<35} {'Metric':<10} {'Before':>9} {'After':>9} {'Delta':>9} {'OK?':>5}")
print("-" * 90)

for _, row in df.iterrows():
    for m in METRICS:
        b_col, a_col = f"before_{m}", f"after_{m}"
        if b_col not in df.columns:
            continue
        b, a = row[b_col], row[a_col]
        d = a - b
        percent = (d / abs(b) * 100) if b != 0 else float('nan')

        if b != 0:
            performance_gain = (a - b)/ abs(b) * 100 if HIGHER_IS_BETTER[m] else (b - a)/ abs(b) * 100
        else:
            performance_gain = float('nan')
            
        improved = (d > 0) if HIGHER_IS_BETTER[m] else (d < 0)
        tag = " ✓" if improved else " ✗"
        print(f"{row['series']:<35} {m:<10} {b:>9.2f} {a:>9.2f} {d:>+9.2f} {performance_gain:>+7.2f}% {tag:>5}")

# Mean row
if len(df) > 1:
    print("-" * 120)
    for m in METRICS:
        b_col, a_col = f"before_{m}", f"after_{m}"
        if b_col not in df.columns:
            continue
        b_mean = df[b_col].mean()
        a_mean = df[a_col].mean()
        b_std = df[b_col].std()
        a_std = df[a_col].std()
        d = a_mean - b_mean
        percent = (d / abs(b_mean) * 100) if b_mean != 0 else float('nan')

        if b_mean != 0:
            performance_gain = (a_mean - b_mean)/ abs(b_mean) * 100 if HIGHER_IS_BETTER[m] else (b_mean - a_mean)/ abs(b_mean) * 100
        else:
            performance_gain = float('nan')

        improved = (d > 0) if HIGHER_IS_BETTER[m] else (d < 0)
        tag = " ✓" if improved else " ✗"
        print(f"{'MEAN':<20} {m:<10} {b_mean:>9.2f} ± {b_std:<8.2f} {a_mean:>9.2f} ± {a_std:<8.2f} {d:>+9.2f}  {performance_gain:>+7.2f}% {tag:>5}")

print(SEP)

# LATEX
NAME_MAP = {
    "Dice": r"\text{DSC}\, (\%)",
    "EAT_Dice": r"\text{DSC}_{\text{EAT}}\, (\%)",
    "NSD": r"\text{NSD}\, (\%)",
    "HD": r"\text{HD}\, (\text{mm})",
    "HD95": r"\text{HD95}\, (\text{mm})",
    "ASSD": r"\text{ASSD}\, (\text{mm})",
    "volume_overlap_inside": r"\text{Internal Violation}\, (\text{cm}^3)",
    "volume_overlap_outside": r"\text{External Violation}\, (\text{cm}^3)",
}

metrics_to_include = ["Dice", "EAT_Dice", "NSD", "HD95", "ASSD", "volume_overlap_inside", "volume_overlap_outside"]

MEDIAN_METRICS = ["volume_overlap_inside", "volume_overlap_outside"]

if len(df) > 1:
    for m in metrics_to_include:
        b_col, a_col = f"before_{m}", f"after_{m}"
        if b_col not in df.columns:
            continue

        higher = HIGHER_IS_BETTER[m]
        arrow = r"$\uparrow$" if higher else r"$\downarrow$"
        name = NAME_MAP.get(m, rf"\text{{{m}}}")

        # -----------------------------
        # CASE 1: MEDIAN + IQR
        # -----------------------------
        if m in MEDIAN_METRICS:
            b_vals = df[b_col].values
            a_vals = df[a_col].values

            b_median = np.median(b_vals)
            a_median = np.median(a_vals)

            b_q25 = np.percentile(b_vals, 25)
            b_q75 = np.percentile(b_vals, 75)

            a_q25 = np.percentile(a_vals, 25)
            a_q75 = np.percentile(a_vals, 75)

            # Gain
            if b_median != 0:
                performance_gain = (
                    (a_median - b_median) / abs(b_median) * 100
                    if higher else
                    (b_median - a_median) / abs(b_median) * 100
                )
            else:
                performance_gain = float('nan')

            # Formatting
            if higher:
                before_str = (
                    rf"\mathbf{{{b_median:.2f}}} \ [{b_q25:.1f}, {b_q75:.1f}]"
                    if b_median >= a_median else
                    rf"{b_median:.2f} \ [{b_q25:.1f}, {b_q75:.1f}]"
                )
                after_str = (
                    rf"\mathbf{{{a_median:.2f}}} \ [{a_q25:.1f}, {a_q75:.1f}]"
                    if a_median > b_median else
                    rf"{a_median:.2f} \ [{a_q25:.1f}, {a_q75:.1f}]"
                )
            else:
                before_str = (
                    rf"\mathbf{{{b_median:.2f}}} \ [{b_q25:.1f}, {b_q75:.1f}]"
                    if b_median <= a_median else
                    rf"{b_median:.2f} \ [{b_q25:.1f}, {b_q75:.1f}]"
                )
                after_str = (
                    rf"\mathbf{{{a_median:.2f}}} \ [{a_q25:.1f}, {a_q75:.1f}]"
                    if a_median < b_median else
                    rf"{a_median:.2f} \ [{a_q25:.1f}, {a_q75:.1f}]"
                )

        # -----------------------------
        # CASE 2: MEAN ± STD (default)
        # -----------------------------
        else:
            b_mean = df[b_col].mean()
            a_mean = df[a_col].mean()
            b_std  = df[b_col].std()
            a_std  = df[a_col].std()

            if b_mean != 0:
                performance_gain = (
                    (a_mean - b_mean) / abs(b_mean) * 100
                    if higher else
                    (b_mean - a_mean) / abs(b_mean) * 100
                )
            else:
                performance_gain = float('nan')

            if higher:
                before_str = (
                    rf"\mathbf{{{b_mean:.2f}}} \pm {b_std:.2f}"
                    if b_mean >= a_mean else
                    rf"{b_mean:.2f} \pm {b_std:.2f}"
                )
                after_str = (
                    rf"\mathbf{{{a_mean:.2f}}} \pm {a_std:.2f}"
                    if a_mean > b_mean else
                    rf"{a_mean:.2f} \pm {a_std:.2f}"
                )
            else:
                before_str = (
                    rf"\mathbf{{{b_mean:.2f}}} \pm {b_std:.2f}"
                    if b_mean <= a_mean else
                    rf"{b_mean:.2f} \pm {b_std:.2f}"
                )
                after_str = (
                    rf"\mathbf{{{a_mean:.2f}}} \pm {a_std:.2f}"
                    if a_mean < b_mean else
                    rf"{a_mean:.2f} \pm {a_std:.2f}"
                )

        # -----------------------------
        # FINAL ROW
        # -----------------------------
        sign = "+" if performance_gain >= 0 else ""

        row = (
            f"& {arrow} ${name}$ "
            f"& ${before_str}$ "
            f"& ${after_str}$ "
            f"& ${sign}{performance_gain:.2f}$\\,\\% \\\\"
        )

        print(row)


# ── 2. Anatomical Violation Table (Percentage Based) ──────────────────
ANATOMICAL_METRICS = {
    "volume_overlap_outside": "External Violation",
    "volume_overlap_inside": "Internal Violation"
}

print("\n% --- ANATOMICAL VIOLATION TABLE ---")
if len(df) > 1:
    for m, display_name in ANATOMICAL_METRICS.items():
        b_col, a_col = f"before_{m}", f"after_{m}"
        if b_col not in df.columns:
            continue

        # 1. Calculate reduction for each individual case
        # Note: (Before - After) / Before gives reduction percentage
        reductions = []
        improved_count = 0
        
        for _, row in df.iterrows():
            b_val, a_val = row[b_col], row[a_col]
            if b_val > 0:
                # Reduction is positive if After < Before
                reduction = (b_val - a_val) / b_val * 100
                reductions.append(reduction)
                if a_val < b_val:
                    improved_count += 1
            elif b_val == 0 and a_val == 0:
                # Already perfect, count as improved/maintained
                improved_count += 1

        if reductions:
            median_reduction = np.median(reductions)
            q25 = np.percentile(reductions, 25)
            q75 = np.percentile(reductions, 75)
        else:
            median_reduction = 0
            q25 = 0
            q75 = 0

        # LaTeX Formatting
        # Formatted as Metric & Median [Q25, Q75] \\
        row_tex = (
            f"& {display_name} "
            f"& {median_reduction:.1f} [{q25:.1f}, {q75:.1f}] \\\\"
        )
        print(row_tex)

print("\n% --- ANATOMICAL VIOLATION TABLE NEW ---")
if len(df) > 1:
    for m, display_name in ANATOMICAL_METRICS.items():
        b_col, a_col = f"before_{m}", f"after_{m}"
        if b_col not in df.columns:
            continue

        # 1. Calculate reduction for each individual case
        # Note: (Before - After) / Before gives reduction percentage
        reductions = []
        improved_count = 0
        
        # Print median before + IQR
        before_values = df[b_col].values
        median_before = np.median(before_values)
        q25_before = np.percentile(before_values, 25)
        q75_before = np.percentile(before_values, 75)
        # print(f"& {display_name} & {median_before:.1f} [{q25_before:.1f}, {q75_before:.1f}] ", end="")

        # median after + IQR
        after_values = df[a_col].values
        median_after = np.median(after_values)
        q25_after = np.percentile(after_values, 25)
        q75_after = np.percentile(after_values, 75)

        # Volume reduction percentage (median of individual reductions)
        performance_gain = ((median_after - median_before) / abs(median_before) * 100) if median_before != 0 else float('nan')
        tag = " ✓" if (median_after < median_before) else " ✗"


        # Print as table
        print(f"{'median':<20} {m:<10} {median_before:>9.2f} [{q25_before:.1f}, {q75_before:.1f}] {median_after:>9.2f} [{q25_after:.1f}, {q75_after:.1f}] {performance_gain:>+7.2f}% {tag:>5}")


import sys
sys.exit(0)

# ── Build tidy dataframe (series | metric | before | after | delta) ────
rows = []
for _, r in df.iterrows():
    for m in METRICS:
        b_col, a_col = f"before_{m}", f"after_{m}"
        if b_col not in df.columns:
            continue
        b, a = r[b_col], r[a_col]
        d = a - b
        rows.append({"series": r["series"], "metric": m, "before": b, "after": a, "delta": d,
                     "improved": (d > 0) if HIGHER_IS_BETTER[m] else (d < 0)})

if len(df) > 1:
    for m in METRICS:
        sub = [r for r in rows if r["metric"] == m]
        if not sub:
            continue
        b = np.mean([r["before"] for r in sub])
        a = np.mean([r["after"]  for r in sub])
        d = a - b
        rows.append({"series": "MEAN", "metric": m, "before": b, "after": a, "delta": d,
                     "improved": (d > 0) if HIGHER_IS_BETTER[m] else (d < 0)})

tidy = pd.DataFrame(rows)


# ── 2. Summary table as image ───────────────────────────────────────────
if len(df) > 1:
    summary = tidy[tidy["series"] == "MEAN"][["metric", "before", "after", "delta"]].copy()
    summary = summary.rename(columns={"metric": "Metric", "before": "Before (mean)",
                                      "after": "After (mean)", "delta": "Δ (mean)"})
else:
    summary = tidy[["metric", "before", "after", "delta"]].copy()
    summary = summary.rename(columns={"metric": "Metric", "before": "Before",
                                      "after": "After", "delta": "Δ"})

mean_delta_total   = (df["after_volume_overlap_outside"]  - df["before_volume_overlap_outside"]).mean()
mean_delta_highres = (df["after_volume_overlap_inside"] - df["before_volume_overlap_inside"]).mean()

extra_rows = pd.DataFrame([
    {"Metric": "Mean Δ Reduction Total",   "Before (mean)": df["before_volume_overlap_outside"].mean(),
     "After (mean)": df["after_volume_overlap_outside"].mean(),   "Δ (mean)": mean_delta_total},
    {"Metric": "Mean Δ Reduction Highres", "Before (mean)": df["before_volume_overlap_inside"].mean(),
     "After (mean)": df["after_volume_overlap_inside"].mean(), "Δ (mean)": mean_delta_highres},
])

if len(df) > 1:
    summary = pd.concat([summary, extra_rows], ignore_index=True)
else:
    extra_rows = extra_rows.rename(columns={"Before (mean)": "Before", "After (mean)": "After", "Δ (mean)": "Δ"})
    summary = pd.concat([summary, extra_rows], ignore_index=True)

# Values are already in correct units — no conversion needed in format_metric_row
def format_metric_row(metric, before, after, delta):
    if metric in ["Dice", "EAT_Dice", "NSD"]:
        return [metric + " (%)", f"{before:.2f}", f"{after:.2f}", f"{delta:+.2f}", "%"]
    elif metric in ["ASD", "ASSD", "HD", "HD95"]:
        return [metric + " (mm)", f"{before:.2f}", f"{after:.2f}", f"{delta:+.2f}", "mm"]
    elif "Δ Reduction Total" in metric or "Δ Reduction Highres" in metric:
        return [metric + " (mm³)", f"{before:.2f}", f"{after:.2f}", f"{delta:+.2f}", "mm³"]
    else:
        return [metric, f"{before:.4f}", f"{after:.4f}", f"{delta:+.4f}", ""]

fig_tbl, ax_tbl = plt.subplots(figsize=(8, 0.6 * len(summary) + 1.2))
ax_tbl.axis("off")
ax_tbl.set_title("Refinement Summary", fontsize=14, fontweight="bold", pad=12)

table = ax_tbl.table(
    cellText=[format_metric_row(r["Metric"], r[summary.columns[1]], r[summary.columns[2]], r[summary.columns[3]])
              for _, r in summary.iterrows()],
    colLabels=["Metric", "Before", "After", "Δ", "Unit"],
    loc="center", cellLoc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

for row_idx, (_, r) in enumerate(summary.iterrows()):
    m = r["Metric"]
    d = float(r[summary.columns[3]])
    if "overlap" in m.lower() or "Δ reduction" in m.lower():
        color = "#C8E6C9" if d > 0 else "#FFCDD2"
    elif HIGHER_IS_BETTER.get(m.replace(" (%)", ""), False):
        color = "#C8E6C9" if d > 0 else "#FFCDD2"
    else:
        color = "#C8E6C9" if d < 0 else "#FFCDD2"
    table[row_idx + 1, 3].set_facecolor(color)

fig_tbl.tight_layout()
path_tbl = os.path.join(out_dir, "summary_table.png")
fig_tbl.savefig(path_tbl, bbox_inches="tight")
print(f"\nSaved: {path_tbl}")


# ── Helper: pick bar colours ────────────────────────────────────────────
def delta_colors(deltas, metric):
    good = "#2E7D32"
    bad  = "#C62828"
    if HIGHER_IS_BETTER[metric]:
        return [good if d > 0 else bad for d in deltas]
    else:
        return [good if d < 0 else bad for d in deltas]


# ── 2. Delta bar chart ──────────────────────────────────────────────────
available = [m for m in METRICS if f"before_{m}" in df.columns]
n = len(available)
ncols = 3
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = np.array(axes).flatten()

for i, m in enumerate(available):
    ax = axes[i]
    deltas = df[f"after_{m}"].values - df[f"before_{m}"].values
    colors = delta_colors(deltas, m)
    bars = ax.bar(x, deltas, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Δ {m}", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=8)
    arrow = "↑ better" if HIGHER_IS_BETTER[m] else "↓ better"
    ax.set_ylabel(f"delta ({arrow})", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for bar, d in zip(bars, deltas):
        offset = 3 if d >= 0 else -12
        ax.annotate(f"{d:+.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, d),
                    xytext=(0, offset), textcoords="offset points",
                    ha="center", fontsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Δ After − Before  (green = improved)", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
path2 = os.path.join(out_dir, "delta_improvement.png")
fig.savefig(path2)
print(f"\nSaved: {path2}")


# ── 3. Before vs After grouped bars ────────────────────────────────────
fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes2 = np.array(axes2).flatten()
bw = 0.35

for i, m in enumerate(available):
    ax = axes2[i]
    b_vals = df[f"before_{m}"].values
    a_vals = df[f"after_{m}"].values
    ax.bar(x - bw / 2, b_vals, bw, label="Before", color="#5B9BD5", edgecolor="white")
    ax.bar(x + bw / 2, a_vals, bw, label="After",  color="#ED7D31", edgecolor="white")
    ax.set_title(m, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

for j in range(i + 1, len(axes2)):
    axes2[j].set_visible(False)

fig2.suptitle("Before vs After Refinement", fontsize=13, fontweight="bold")
fig2.tight_layout(rect=[0, 0, 1, 0.95])
path3 = os.path.join(out_dir, "before_vs_after.png")
fig2.savefig(path3)
print(f"Saved: {path3}")


# ── 4. Normalised improvement boxplot ──────────────────────────────────
# Dice/EAT_Dice/NSD are already in % (0-100), so normalise accordingly
norm_scores = {}
for m in available:
    before = df[f"before_{m}"].values.astype(float)
    after  = df[f"after_{m}"].values.astype(float)
    if m in ["Dice", "EAT_Dice", "NSD"]:
        # values are 0-100, normalise against 100-point scale
        if HIGHER_IS_BETTER[m]:
            score = (after - before) / np.maximum(100 - before, 1e-8)
        else:
            score = (before - after) / np.maximum(before, 1e-8)
    else:
        if HIGHER_IS_BETTER[m]:
            score = (after - before) / np.maximum(1 - before, 1e-8)
        else:
            score = (before - after) / np.maximum(before, 1e-8)
    norm_scores[m] = np.clip(score, -1, 1)

fig3, ax3 = plt.subplots(figsize=(8, 5))
bp = ax3.boxplot(
    [norm_scores[m] for m in available],
    labels=available,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=1.5)
)
for patch in bp["boxes"]:
    patch.set_facecolor("#5B9BD5")
ax3.axhline(0, color="black", linestyle="--", linewidth=1)
ax3.set_ylabel("Normalised improvement  (> 0 = better)")
ax3.set_title("Normalised improvement across metrics", fontweight="bold")
ax3.grid(axis="y", alpha=0.3)
fig3.tight_layout()
path4 = os.path.join(out_dir, "normalised_improvement_boxplot.png")
fig3.savefig(path4)
print(f"Saved: {path4}")

plt.show()
print("\nDone.")