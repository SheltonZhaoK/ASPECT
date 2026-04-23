import json, matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rcParams['svg.fonttype'] = 'none'

def plot_temporal_trends_with_cancer(results):
    rows = []
    for r in results:
        stats = r.get("rule_stats", {}) or {}
        freq = stats.get("overall_frequency")
        start_date = stats.get("start_date_mean")
        dataset = r.get("dataset", "unknown")

        if not freq or not start_date:
            continue

        try:
            year = int(str(start_date)[:4])
        except Exception:
            continue

        rows.append({
            "year": year,
            "dataset": dataset,
            "frequency": freq
        })

    df = pd.DataFrame(rows)

    # ✅ Average frequency across rule types (inclusion + exclusion)
    df_yearly = (
        df.groupby(["year", "dataset"])["frequency"]
          .mean()
          .reset_index(name="avg_freq")
    )

    # Custom color palette per cancer type
    palette = {
        "breast_cancer": "#28A9A1",
        "colorectal_cancer": "#F4A016",
        "lung_cancer": "#E71F19",
        "melanoma": "#7262ac",
        "prostate_cancer": "#E9BD27"
    }

    plt.figure(figsize=(10, 6))
    for dataset, color in palette.items():
        sub = df_yearly[df_yearly["dataset"] == dataset]
        if not sub.empty:
            sub = sub.sort_values("year")
            plt.plot(
                sub["year"],
                sub["avg_freq"],
                linestyle="-",
                marker="o",
                markersize=5,
                color=color,
                label=dataset.replace("_", " ").title()
            )

    plt.title("Temporal Trends of Eligibility Rule Frequency (Inclusion + Exclusion Combined)")
    plt.xlabel("Trial Start Year")
    plt.ylabel("Average Rule Frequency (0–1)")
    plt.legend(
        loc="lower center",
        ncol=3,
        frameon=False
    )
    plt.tight_layout()
    plt.savefig("../results/frequency_trends_combined_avg.svg")
    plt.savefig("../results/frequency_trends_combined_avg.png")
    plt.close()

def plot_temporal_enrollment(results, palette=None):
    """
    Temporal trends in enrollment.
    - Average EPSM (enrollment per site per month) aggregated by year and dataset
    """

    rows = []
    for r in results:
        stats = r.get("rule_stats", {}) or {}
        epsm = stats.get("epsm_mean")  # <-- back to EPSM
        start_date = stats.get("start_date_mean")
        trial_id = r["trial_id"]
        dataset = r["dataset"]

        if not epsm or not start_date:
            continue

        try:
            year = int(str(start_date)[:4])  # extract year from YYYY-MM-DD
        except Exception:
            continue

        rows.append({
            "year": year,
            "dataset": dataset,
            "trial_id": trial_id,
            "epsm": epsm
        })

    df = pd.DataFrame(rows)

    # ---- Aggregate ----
    # Average EPSM per trial-year-dataset
    epsm_yearly = (
        df.groupby(["year", "dataset"])["epsm"]
          .mean()
          .reset_index()
    )

    # ---- Default palette if none given ----
    if palette is None:
        palette = {
            "breast_cancer": "#28A9A1",
            "colorectal_cancer": "#F4A016",
            "lung_cancer": "#E71F19",
            "melanoma": "#7262ac",
            "prostate_cancer": "#E9BD27"
        }

    # ---- Plot ----
    plt.figure(figsize=(11, 7))

    sns.lineplot(
        data=epsm_yearly,
        x="year", y="epsm",
        hue="dataset",
        marker="o",
        linewidth=2,
        palette=palette,
        hue_order=list(palette.keys())  # enforce consistent color order
    )

    plt.title("Temporal Trends in EPSM by Cancer Type")
    plt.xlabel("Trial Start Year")
    plt.ylabel("Average EPSM (enrollment per site per month)")

    plt.tight_layout()
    plt.savefig("../results/temporal_enrollment_epsm.svg", bbox_inches="tight")
    plt.savefig("../results/temporal_enrollment_epsm.png", bbox_inches="tight")

def plot_frequency_vs_epsm(results):
    """
    Scatterplot of rule frequency vs. EPSM across years.
    Each point = one rule (inclusion or exclusion).
    Color = trial start year.
    """

    rows = []
    for r in results:
        stats = r.get("rule_stats", {}) or {}
        freq = stats.get("overall_frequency")
        epsm = stats.get("epsm_mean")
        start_date = stats.get("start_date_mean")

        if freq is None or epsm is None or not start_date:
            continue

        try:
            year = int(str(start_date)[:4])
        except Exception:
            continue

        rows.append({
            "frequency": freq,
            "epsm_mean": epsm,
            "year": year,
            "dataset": r.get("dataset", "unknown"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("⚠️ No valid data found for frequency–EPSM plot.")
        return

    # ---- Optional: filter extreme outliers for clearer scatter ----
    df = df[(df["epsm_mean"].notna()) & (df["epsm_mean"] > 0) & (df["epsm_mean"] < df["epsm_mean"].quantile(0.99))]

    # ---- Plot ----
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=df,
        x="frequency",
        y="epsm_mean",
        color="#28A9A1",
        alpha=0.8,
        edgecolor="none"
    )

    plt.title("Rule Frequency vs. EPSM Across Years")
    plt.xlabel("Rule Frequency (0–1)")
    plt.ylabel("Average EPSM (Enrollment per Site per Month)")

    plt.tight_layout()
    plt.savefig("../results/frequency_vs_epsm_new.svg", bbox_inches="tight")
    plt.savefig("../results/frequency_vs_epsm_new.png", bbox_inches="tight")
    plt.close()

def plot_temporal_combined(results):
    """
    Combines:
    1. Temporal trends of eligibility rule frequency (inclusion + exclusion combined)
    2. Temporal trends in EPSM (enrollment per site per month)
    """

    # --- Custom color palette ---
    palette = {
        "breast_cancer": "#28A9A1",
        "colorectal_cancer": "#F4A016",
        "lung_cancer": "#E71F19",
        "melanoma": "#7262ac",
        "prostate_cancer": "#E9BD27"
    }

    # =========================================================
    # (1) Prepare rule frequency data
    # =========================================================
    freq_rows = []
    for r in results:
        stats = r.get("rule_stats", {}) or {}
        freq = stats.get("overall_frequency")
        start_date = stats.get("start_date_mean")
        dataset = r.get("dataset", "unknown")

        if not freq or not start_date:
            continue
        try:
            year = int(str(start_date)[:4])
        except Exception:
            continue

        freq_rows.append({
            "year": year,
            "dataset": dataset,
            "frequency": freq
        })

    freq_df = pd.DataFrame(freq_rows)
    freq_yearly = (
        freq_df.groupby(["year", "dataset"])["frequency"]
        .mean()
        .reset_index(name="avg_freq")
    )

    # =========================================================
    # (2) Prepare EPSM data
    # =========================================================
    epsm_rows = []
    for r in results:
        stats = r.get("rule_stats", {}) or {}
        epsm = stats.get("epsm_mean")
        start_date = stats.get("start_date_mean")
        trial_id = r.get("trial_id")
        dataset = r.get("dataset")

        if not epsm or not start_date:
            continue
        try:
            year = int(str(start_date)[:4])
        except Exception:
            continue

        epsm_rows.append({
            "year": year,
            "dataset": dataset,
            "trial_id": trial_id,
            "epsm": epsm
        })

    epsm_df = pd.DataFrame(epsm_rows)
    epsm_yearly = (
        epsm_df.groupby(["year", "dataset"])["epsm"]
        .mean()
        .reset_index()
    )

    # =========================================================
    # (3) Plot side-by-side subplots
    # =========================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=False)

    # --- Left: Rule Frequency ---
    ax1 = axes[0]
    for dataset, color in palette.items():
        sub = freq_yearly[freq_yearly["dataset"] == dataset]
        sub = sub.sort_values("year")
        sub = sub.dropna(subset=["avg_freq"])
        if not sub.empty:
            plt.sca(ax1)
            plt.plot(
                sub["year"],
                sub["avg_freq"],
                linestyle="-",
                marker="o",
                markersize=5,
                color=color,
                label=dataset.replace("_", " ").title(),
                solid_capstyle="round"
            )

    ax1.set_title("Temporal Trends of Eligibility Rule Frequency", fontsize=12)
    ax1.set_xlabel("Trial Start Year")
    ax1.set_ylabel("Average Rule Frequency (0–1)")
    sns.despine(ax=ax1)
    ax1.grid(False)

    # --- Right: EPSM ---
    ax2 = axes[1]
    for dataset, color in palette.items():
        sub = epsm_yearly[epsm_yearly["dataset"] == dataset]
        sub = sub.sort_values("year")
        sub = sub.dropna(subset=["epsm"])

        if not sub.empty:
            plt.sca(ax2)
            plt.plot(
                sub["year"],
                sub["epsm"],
                linestyle="-",
                marker="o",
                markersize=5,
                color=color,
                solid_capstyle="round"
            )

    ax2.set_title("Temporal Trends in EPSM", fontsize=12)
    ax2.set_xlabel("Trial Start Year")
    ax2.set_ylabel("Average EPSM (enrollment per site per month)")
    sns.despine(ax=ax2)
    ax2.grid(False)

    # --- Unified legend and layout ---
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, title="", loc="lower center", ncol=3, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig("../results/temporal_combined.svg", bbox_inches="tight")
    plt.savefig("../results/temporal_combined.png", bbox_inches="tight")

if __name__ == "__main__":
    # Example: load your results JSON
    json_file = "/home/konghaoz/trialDesigner/results/full_rule_summary_nli_check_0.6.json"
    with open(json_file, "r") as f:
        data = json.load(f)

    results = []
    for dataset, dataset_data in data.items():
        trials = dataset_data.get("trials", {})
        for trial_id, trial_data in trials.items():
            for rule_type, rules in trial_data.items():  # inclusion / exclusion
                for rule_name, rule_content in rules.items():
                    rule_stats = rule_content.get("rule_stats", {})
                    results.append({
                        "dataset": dataset,
                        "trial_id": trial_id,
                        "rule_type": rule_type,
                        "rule_name": rule_name,
                        "rule_stats": rule_stats,
                    })

    # plot_temporal_trends_with_cancer(results)
    plot_frequency_vs_epsm(results)
    # plot_temporal_enrollment(results, palette={
    # "breast_cancer": "#28A9A1",
    # "colorectal_cancer": "#F4A016",
    # "lung_cancer": "#E71F19",
    # "melanoma": "#7262ac",
    # "prostate_cancer": "#E9BD27"})
    # plot_temporal_combined(results)