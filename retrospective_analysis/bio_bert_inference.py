import argparse, torch, os, sys, json, matplotlib
from pathlib import Path
import pandas as pd
import numpy as np
import torch.nn.functional as F
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from transformers import AutoTokenizer, AutoModelForSequenceClassification
matplotlib.rcParams['svg.fonttype'] = 'none'

# Paths
BASE = Path("/home/konghaoz/activeLearning_trialRecruit/results")
OUT  = BASE / "caseStudy_LungCancer";  OUT.mkdir(parents=True, exist_ok=True)

def _join(h, items): return f"{h}:\n" + "\n".join(f"- {s.strip()}" for s in items) if items else f"{h}:"


def _score_cls(model, tokenizer, inc, exc, device):
    """Compute P(label=1) for a single inclusion/exclusion text pair."""
    model.eval()
    enc = tokenizer(
        inc,
        text_pair=exc,
        truncation=True,
        padding="longest",
        max_length=512,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
        prob = F.softmax(logits, dim=1)[0, 1].item()
    return prob

@torch.no_grad()
def ablate_cls(model, tokenizer, inc_list, exc_list, device):
    """Ablate each inclusion/exclusion rule and measure impact on classification."""
    inc_full = " ".join(["inclusion criteria:"] + inc_list)
    exc_full = " ".join(["exclusion criteria:"] + exc_list)
    
    # baseline full prediction
    full_prob = _score_cls(model, tokenizer, inc_full, exc_full, device)

    rows = []
    # # ablate inclusion rules
    # for i in range(len(inc_list)):
    #     inc_ablated = " ".join(["inclusion criteria:"] + inc_list[:i] + inc_list[i+1:])
    #     p = _score_cls(model, tokenizer, inc_ablated, exc_full, device)
    #     rows.append(dict(which="inclusion", idx=i, delta=p - full_prob, rule_name=inc_list[i]))

    # # ablate exclusion rules
    # for j in range(len(exc_list)):
    #     exc_ablated = " ".join(["exclusion criteria:"] + exc_list[:j] + exc_list[j+1:])
    #     p = _score_cls(model, tokenizer, inc_full, exc_ablated, device)
    #     rows.append(dict(which="exclusion", idx=j, delta=p - full_prob, rule_name=exc_list[j]))
    
    return full_prob, pd.DataFrame(rows)

@torch.no_grad()
def load_model_with_stats(model_dir, device="cuda"):
    model  = AutoModelForSequenceClassification.from_pretrained(f"{model_dir}/model_10e").to(device)
    with open(Path(model_dir) / "label_stats.json") as f:
        stats = json.load(f)                       # {'mu': ..., 'sd': ...}
    return model, stats["mu"], stats["sd"]

def ablate_reg(model, tok, mu, sd, inc, exc, dev="cuda"):
    def _raw(i, e):
        batch = tok(i, text_pair=e, truncation=True, padding="longest",
                     max_length=512, return_tensors="pt").to(dev)
        return model(**batch).logits.squeeze().item()   # z‑score

    inc_full, exc_full = _join("inclusion criteria", inc), _join("exclusion criteria", exc)

    full_z = _raw(inc_full, exc_full)
    full   = full_z * sd + mu                           # back‑transform

    rows = []
    for i in range(len(inc)):
        p   = _raw(_join("inclusion criteria", inc[:i]+inc[i+1:]), exc_full)
        rows.append({"which": "inclusion", "idx": i, "delta": (p - full_z) * sd})

    for j in range(len(exc)):
        p   = _raw(inc_full, _join("exclusion criteria", exc[:j]+exc[j+1:]))
        rows.append({"which": "exclusion", "idx": j, "delta": (p - full_z) * sd})

    return full, pd.DataFrame(rows)

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    output = []
    for data in ["breast_cancer", "prostate_cancer", "colorectal_cancer", "melanoma", "lung_cancer"]:
        # trial = 'NCT02566993' #NCT03098030
        inclusion_trial_data = pd.read_csv(f"../data/application/{data}/structured_trial_data_gpt-4.1-2025-04-14_inclusion_application.csv")
        exclusion_trial_data = pd.read_csv(f"../data/application/{data}/structured_trial_data_gpt-4.1-2025-04-14_exclusion_application.csv")
        trial_id_list = inclusion_trial_data["Trial_ID"].to_list()
        for trial in trial_id_list:
            cur_trial_inclusion = inclusion_trial_data[inclusion_trial_data["Trial_ID"] == trial].dropna(axis=1, how='all')
            cur_trial_inclusion = cur_trial_inclusion.drop("Trial_ID", axis=1)
            cur_trial_inclusion.columns = [col.split(": ")[1].lower() if ": " in col else col.lower() for col in cur_trial_inclusion.columns]

            cur_trial_exclusion = exclusion_trial_data[exclusion_trial_data["Trial_ID"] == trial].dropna(axis=1, how='all')
            cur_trial_exclusion = cur_trial_exclusion.drop("Trial_ID", axis=1)
            cur_trial_exclusion.columns = [col.split(": ")[1].lower() if ": " in col else col.lower() for col in cur_trial_exclusion.columns]

            INCLUSION_CRITERIA = []
            for _, row in cur_trial_inclusion.iterrows():
                    for col, val in row.items():
                        if val == "descriptive":
                            rule_str = f"{col}"
                        else:
                            rule_str = f"{col}: {val}"
                        INCLUSION_CRITERIA.append(rule_str)
            
            EXCLUSION_CRITERIA = []
            for _, row in cur_trial_exclusion.iterrows():
                for col, val in row.items():
                    if val == "descriptive":
                        rule_str = f"{col}"
                    else:
                        rule_str = f"{col}: {val}"
                    EXCLUSION_CRITERIA.append(rule_str)

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            tok = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

            # ── classification ablations (mortality / adverse_event) ────────────────
            apps = ["mortality", "adverse_event"]
            # apps = ["adverse_event"]
            cls_data = []
            full_probs = {}
            for app in apps:
                mdl = AutoModelForSequenceClassification.from_pretrained(BASE / app / f"phase3").to(dev)
                full, df = ablate_cls(mdl, tok, INCLUSION_CRITERIA, EXCLUSION_CRITERIA, dev)
                if full < 0.9:
                    a = f"{data} | {trial} | {full}\n"
                    print(a)
                    output.append(a)
        print(output)
    #     df["app"] = app
    #     cls_data.append(df)
    #     full_probs[app] = full
    # df_cls = pd.concat(cls_data, ignore_index=True)
    # df_cls.to_csv("/home/konghaoz/activeLearning_trialRecruit/results/caseStudy_LungCancer/cls_ablation.csv")

    # # ── regression ablation (Enrollment, Site Count, Months, EPSM) ────────────────
    # reg_targets = ["Enrollment Count", "Site Count", "Months", "EPSM"]
    # reg_data = []
    # full_vals = {}

    # for target in reg_targets:
    #     model_dir = f"{BASE}/{target}"
    #     model, mu, sd = load_model_with_stats(model_dir)   
    #     full, df = ablate_reg(model, tok, mu, sd, INCLUSION_CRITERIA, EXCLUSION_CRITERIA, dev)
    #     df["target"] = target
    #     df.to_csv(f"/home/konghaoz/activeLearning_trialRecruit/results/caseStudy_LungCancer/{target}_ablation.csv")
    #     reg_data.append(df)
    #     full_vals[target] = full

    #     # ── print classification baselines ────────────────────────────────
    # print("\n🧩 Classification Baselines (P(label=1) with all rules):")
    # for app, val in full_probs.items():
    #     print(f"  {app:<20}: {val:.4f}")

    # # ── print regression baselines ────────────────────────────────────
    # print("\n📈 Regression Baselines (Predicted value with all rules):")
    # for target, val in full_vals.items():
    #     print(f"  {target:<20}: {val:.4f}")
    
    # cls_long = df_cls.copy()
    # cls_long.rename(columns={"app": "metric"}, inplace=True)
    # reg_long = pd.concat(reg_data, ignore_index=True)
    # reg_long.rename(columns={"target": "metric"}, inplace=True)

    # df_all = pd.concat([cls_long, reg_long], ignore_index=True)
    # df_all["criterion"] = df_all["idx"] + 1  # 1-based index for display
    # df_all["row_label"] = df_all["metric"] + " – " + df_all["which"]

    # # Pivot: rows = metric+type, columns = criterion, values = delta
    # pivot = df_all.pivot_table(index="row_label", columns="criterion", values="delta")

    # cmap = sns.diverging_palette(240, 10, as_cmap=True)

    # plt.figure(figsize=(18, 9))
    # ax = sns.heatmap(
    #     pivot,
    #     cmap=cmap,
    #     center=0,                # baseline = white
    #     annot=True,
    #     fmt=".2f",
    #     linewidths=0.5,
    #     cbar_kws={'label': 'Δ from baseline'},
    # )

    # ax.set_title("Ablation Study: Criterion-wise Δ from Baseline", fontsize=18, pad=15)
    # ax.set_xlabel("Criterion Index", fontsize=13)
    # ax.set_ylabel("Metric – Type", fontsize=13)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    # ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    # plt.tight_layout()

    # out_png = OUT / "ablation_heatmap_teal_white_gold.png"
    # out_png = OUT / "ablation_heatmap_teal_white_gold.svg"
    # plt.savefig(out_png, dpi=250, bbox_inches="tight")
    # print(f"✓ wrote {out_png}")

if __name__ == "__main__":
    main()
