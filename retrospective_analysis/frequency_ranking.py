import json,os,re,umap,hdbscan,matplotlib
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from collections import defaultdict
from sentence_transformers import SentenceTransformer

import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.fonttype'] = 'none'

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
def print_top5_summary(top_rules):
    for dataset, rules in top_rules.items():
        print(f"\n=== {dataset.upper()} ===")
        print("Top 5 most frequent rules:\n")

        for r in rules[:10]:
            stats = r.get("rule_stats", {}) or {}
            rule_type = r.get("rule_type", "NA").capitalize()  # Inclusion / Exclusion

            print(f"[{rule_type}] {r['rule_name']}")

            def fmt(val):
                if isinstance(val, (int, float)):
                    return f"{val:.3f}"
                return val

            print(
                f"Freq={fmt(stats.get('overall_frequency', 'NA'))}, "
                f"#Trials={fmt(stats.get('n_trials', 'NA'))}, "
                f"AveEnroll={fmt(stats.get('enrollment_mean', 'NA'))}, "
                f"AveSites={fmt(stats.get('site_mean', 'NA'))}, "
                f"AveRecruitLen={fmt(stats.get('recruitment_months_mean', 'NA'))}, "
                f"AveEPSM={fmt(stats.get('epsm_mean', 'NA'))}, "
                f"AveStartDate={stats.get('start_date_mean', 'NA')}"
            )
            print("-" * 80)

def summarize_rules_batch(rule_names, model="gpt-5"):
    unique_rules = list(dict.fromkeys(rule_names))  # preserve order, remove duplicates

    prompt = (
        "Summarize the following clinical trial eligibility rules into GENERIC categories "
        "that unify similar concepts across cancer types. Examples:\n"
        "- 'Prior Chemotherapy', 'Prior Prostate Cancer Therapy' → 'Prior Therapy'\n"
        "- 'Breast Cancer Diagnosis', 'Melanoma Diagnosis' → 'Cancer Diagnosis'\n"
        "- 'Breast Cancer Stage', 'Resected Melanoma Stage' → 'Cancer Stage'\n"
        "\n⚠️ IMPORTANT: Return ONLY a valid JSON array (list), in the SAME order as the rules are listed, "
        "with each element being the normalized category string. No keys, no extra text.\n\n"
        "Rules:\n" + "\n".join(f"{i+1}. {rule}" for i, rule in enumerate(unique_rules))
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = resp.choices[0].message.content.strip()
    print(raw)
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)   # remove opening fence
        raw = re.sub(r"\n?```$", "", raw)            # remove closing fence
        raw = raw.strip()

    try:
        summary_list = json.loads(raw)
        if not isinstance(summary_list, list):
            raise ValueError("Not a list")
    except Exception:
        print("⚠️ Could not parse GPT output as JSON list. Raw output:\n", raw)
        summary_list = ["Other"] * len(unique_rules)

    used_counts = defaultdict(int)
    final_summaries = {}
    for rule_name, summary in zip(unique_rules, summary_list):
        used_counts[summary] += 1
        if used_counts[summary] > 1:
            summary = f"{summary}_{used_counts[summary]}"
        final_summaries[rule_name] = summary

    return final_summaries
    
def get_top_rules(results, top_k=10):
    """Return the top_k most frequent (unique) rules for each dataset."""
    rows = []
    for r in results:
        freq = r["rule_stats"].get("overall_frequency")
        n_trials = r["rule_stats"].get("n_trials")
        rows.append({
            "dataset": r["dataset"],
            "rule_type": r["rule_type"],
            "rule_name": r["rule_name"],
            "overall_frequency": freq,
            "n_trials": n_trials,
            "rule_stats": r["rule_stats"]
        })

    df_rules = pd.DataFrame(rows)

    top_rules = {}
    for dataset, subdf in df_rules.groupby("dataset"):
        subdf = (
            subdf.sort_values("overall_frequency", ascending=False)
                 .drop_duplicates(subset=["rule_name", "rule_type"], keep="first")
        )
        # 👇 don't overwrite the function argument
        top_rules_df = subdf.head(top_k)
        top_rules[dataset] = top_rules_df.to_dict(orient="records")

    return top_rules

def plot_rule_categories(top_rules, summaries):
    """
    Plot all datasets' top rules in ONE figure.
    First 10 bars = dataset A, next 10 = dataset B, etc.
    Bars are colored by dataset.
    """
    rows = []
    for dataset, rules in top_rules.items():
        for r in rules:
            cat = summaries.get(r["rule_name"], "Other")
            rows.append({
                "dataset": dataset,
                "category": cat,
                "overall_frequency": r["overall_frequency"]
            })

    df = pd.DataFrame(rows)

    # Build category order
    ordered = []
    for dataset in top_rules.keys():
        sub = df[df["dataset"] == dataset].sort_values("overall_frequency", ascending=False)
        ordered.extend(sub["category"].tolist())

    # Deduplicate while preserving order
    seen = set()
    ordered_unique = []
    for cat in ordered:
        if cat not in seen:
            ordered_unique.append(cat)
            seen.add(cat)

    df["category"] = pd.Categorical(df["category"], categories=ordered_unique[::-1], ordered=True)

    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=df,
        x="overall_frequency",
        y="category",
        hue="dataset",      # color by dataset
        dodge=False,        # stack y categories, not side-by-side
        palette=["#28A9A1", "#F4A016", "#E71F19", "#7262ac", "#E9BD27", "#7D8995", "#C9A77C", "#F6BBC6", "#D6594C"]
    )
    plt.title("Top 10 Rule Categories per Dataset (Combined)")
    plt.xlabel("Frequency")
    plt.ylabel("Rule Category")
    plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("../results/frequency_analysis_combined.svg", bbox_inches="tight")
    plt.savefig("../results/frequency_analysis_combined.png", bbox_inches="tight")


def select_top_unique_rules_with_gpt(top_rules, k=10):
    """
    For each dataset (rules sorted by frequency):
      1) keep the first rule,
      2) for each next rule, compare with ALL already kept rules using GPT,
      3) include only if GPT says NO for all comparisons,
      4) stop when k rules are kept.
    Returns dict[dataset] -> list of rule dicts (same format as top_rules).
    """
    def is_same_rule(a, b):
        prompt = (
            "You are comparing two clinical trial eligibility rules.\n\n"
            "Task: Decide if they describe the similar aspects for example both talking about history or both talking about prior treatment.\n"
            "- Answer YES if they describe similar theme (e.g., 'history', 'performance status').\n"
            "- Small differences in description timeframe, stage, or exceptions DO NOT matter.\n"
            "- Answer NO if they are about different concepts (e.g., one about malignancy history vs one about renal function).\n\n"
            "Examples:\n"
            "Rule A: History of malignancy within 5 years except carcinoma in situ\n"
            "Rule B: History of non-breast malignancies within 3 years\n"
            "Answer: YES\n\n"
            "Rule A: Histologically confirmed adenocarcinoma of the breast\n"
            "Rule B: Cytologically proven diagnosis of adenocarcinoma of the breast\n"
            "Answer: YES\n\n"
            "Rule A: History of malignancy within 5 years\n"
            "Rule B: ECOG performance status ≤ 2\n"
            "Answer: NO\n\n"
            f"Rule A:\n{a}\n\nRule B:\n{b}\n\nAnswer:"
        )
        resp = client.chat.completions.create(
            model="gpt-4.1",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        ans = resp.choices[0].message.content.strip().upper()
        return ans.startswith("Y")

    out = {}
    for dataset, rules in top_rules.items():
        kept, kept_texts = [], []
        for r in rules:
            if len(kept) >= k:
                break
            candidate = r["rule_name"]

            # skip exact duplicates
            if any(candidate.lower() == t.lower() for t in kept_texts):
                continue

            # compare against ALL previously kept rules
            is_duplicate = False
            for t in kept_texts:
                try:
                    if is_same_rule(candidate, t):
                        is_duplicate = True
                        break
                except Exception:
                    # if GPT call fails, assume not duplicate
                    is_duplicate = False

            if not is_duplicate:
                kept.append(r)  # keep full dict
                kept_texts.append(candidate)

        out[dataset] = kept
    return out

def load_cached_top_rules_and_summaries():
    with open("../results/top_rules.json", "r") as f:
        top_rules = json.load(f)
    with open("../results/summaries.json", "r") as f:
        summaries = json.load(f)
    return top_rules, summaries


def main(use_cache=True):
    if use_cache:
        top_rules, summaries = load_cached_top_rules_and_summaries()
        if top_rules and summaries:
            print("✅ Loaded cached top rules and summaries.")
            # plot_rule_categories(top_rules, summaries)
            print_top5_summary(top_rules)
            return

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

    base_top_rules_100 = get_top_rules(results, top_k=100)
    top_rules = select_top_unique_rules_with_gpt(base_top_rules_100)
    all_rule_names = []
    for dataset, rules in top_rules.items():
        all_rule_names.extend([r["rule_name"] for r in rules])
    summaries = summarize_rules_batch(all_rule_names)

    with open("../results/top_rules.json", "w") as f:
        json.dump(top_rules, f, indent=2)
    with open("../results/summaries.json", "w") as f:
        json.dump(summaries, f, indent=2)

    plot_rule_categories(top_rules, summaries)
    print_top5_summary(top_rules)



if __name__ == "__main__":
    main()