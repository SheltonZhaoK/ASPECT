import json, matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
matplotlib.rcParams['svg.fonttype'] = 'none'


# ========================
# 1. Build trial-level dataset
# ========================

def build_trial_level_df(results):
    rows = []
    for r in results:
        stats = r.get("rule_stats", {}) or {}
        trial_id = r["trial_id"]
        dataset = r["dataset"]
        rule_type = r["rule_type"].lower()  # inclusion/exclusion
        start_date = stats.get("start_date_mean")

        if not start_date:
            continue
        try:
            year = int(str(start_date)[:4])
        except Exception:
            year = None

        rows.append({
            "trial_id": trial_id,
            "dataset": dataset,
            "rule_type": rule_type,
            "epsm": stats.get("epsm_mean"),
            "enrollment": stats.get("enrollment_mean"),
            "sites": stats.get("site_mean"),
            "year": year
        })

    df = pd.DataFrame(rows)

    # ---- Aggregate to trial-level ----
    trial_level = (
        df.groupby(["trial_id", "dataset", "year"])
          .agg(
              epsm=("epsm", "mean"),
              enrollment=("enrollment", "mean"),
              sites=("sites", "mean"),
              n_rules=("rule_type", "count"),
              n_incl=("rule_type", lambda x: (x == "inclusion").sum()),
              n_excl=("rule_type", lambda x: (x == "exclusion").sum())
          )
          .reset_index()
    )

    # Add exclusion percentage
    trial_level["p_excl"] = trial_level["n_excl"] / trial_level["n_rules"]

    return trial_level


# ========================
# 2. Plotting
# ========================

def plot_epsm_heatmaps(trial_level):
    # Drop rows with missing values
    df = trial_level.dropna(subset=["n_rules", "p_excl", "epsm"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Build a colormap from your palette base color (#28A9A1)
    base_color = "#28A9A1"
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap", ["#e0f7f5", base_color, "#003c38"]
    )

    # ---- Panel A: n_rules vs EPSM ----
    hb1 = axes[0].hist2d(
        df["n_rules"], df["epsm"],
        bins=[20, 30], cmap=cmap,
        norm=mcolors.LogNorm()
    )
    axes[0].set_title("Trial Complexity vs EPSM (Density)")
    axes[0].set_xlabel("Total # of Rules (n_rules)")
    axes[0].set_ylabel("Enrollment per Site per Month (EPSM)")
    cb1 = fig.colorbar(hb1[3], ax=axes[0])
    cb1.set_label("log(Trial Count)")

    # ---- Panel B: p_excl vs EPSM ----
    hb2 = axes[1].hist2d(
        df["p_excl"], df["epsm"],
        bins=[20, 30], cmap=cmap,
        norm=mcolors.LogNorm()
    )
    axes[1].set_title("Exclusion Share vs EPSM (Density)")
    axes[1].set_xlabel("Proportion of Exclusion Rules (p_excl)")
    axes[1].set_ylabel("")
    cb2 = fig.colorbar(hb2[3], ax=axes[1])
    cb2.set_label("log(Trial Count)")

    plt.tight_layout()
    plt.savefig("../results/epsm_heatmaps_singlecolor.png", dpi=300)
    plt.savefig("../results/epsm_heatmaps_singlecolor.svg")

def plot_restrictiveness_vs_epsm(trial_level):
    palette = {
        "breast_cancer": "#28A9A1",
        "colorectal_cancer": "#F4A016",
        "lung_cancer": "#E71F19",
        "melanoma": "#7262ac",
        "prostate_cancer": "#E9BD27"
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # ---- Panel A: Complexity (n_rules vs EPSM) ----
    sns.scatterplot(
        data=trial_level, x="n_rules", y="epsm",
        hue="dataset", alpha=0.7, palette=palette, ax=axes[0]
    )
    axes[0].set_title("Trial Complexity vs EPSM")
    axes[0].set_xlabel("Total # of Rules (n_rules)")
    axes[0].set_ylabel("Enrollment per Site per Month (EPSM)")

    # ---- Panel B: Restrictiveness (% exclusion vs EPSM) ----
    sns.scatterplot(
        data=trial_level, x="p_excl", y="epsm",
        hue="dataset", alpha=0.7, palette=palette, ax=axes[1]
    )
    axes[1].set_title("Exclusion Rule Share vs EPSM")
    axes[1].set_xlabel("Proportion of Exclusion Rules (p_excl)")
    axes[1].set_ylabel("")

    # Legend once, outside the plots
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, title="Cancer Type",
               bbox_to_anchor=(1.02, 0.5), loc="center left")

    plt.tight_layout()
    plt.savefig("../results/restrictiveness_vs_epsm.png")
    plt.savefig("../results/restrictiveness_vs_epsm.svg")


# ========================
# 3. Main
# ========================

def main():
    # Load your full JSON results file
    json_file = "/home/konghaoz/trialDesigner/results/full_rule_summary_nli_check_0.6.json"
    with open(json_file, "r") as f:
        data = json.load(f)

    results = []
    for dataset, dataset_data in data.items():
        trials = dataset_data.get("trials", {})
        for trial_id, trial_data in trials.items():
            for rule_type, rules in trial_data.items():
                for rule_name, rule_content in rules.items():
                    rule_stats = rule_content.get("rule_stats", {})
                    results.append({
                        "dataset": dataset,
                        "trial_id": trial_id,
                        "rule_type": rule_type,
                        "rule_name": rule_name,
                        "rule_stats": rule_stats,
                    })

    trial_level = build_trial_level_df(results)
    plot_restrictiveness_vs_epsm(trial_level)
    plot_epsm_heatmaps(trial_level)


if __name__ == "__main__":
    main()