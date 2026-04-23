import json
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.metrics.pairwise import cosine_distances

# ============================================================
# Paths & config
# ============================================================
JSON_PATH = Path("../results/full_rule_summary_nli_check_0.6.json")
CSV_PATH = Path("../results/all_trials_rule_exclusion_rates.csv")
CLASSIFIED_JSON = Path("../results/classified_interventions.json")
OUTPUT_PATH = Path("../results/merged_rule_level_summary_final.csv")

MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"

EXCLUSION_COLS = [
    "Overall",
    "White",
    "Asian",
    "African-American",
    "Female",
    "Male",
    "18-50",
    "50-65",
    ">65",
]

DRUG_COLUMNS = [
    "Drug: Chemotherapy",
    "Drug: Targeted Therapy",
    "Drug: Immunotherapy / Biological Therapy",
    "Drug: Hormonal Therapy",
    "Drug: Photodynamic Therapy",
    "Drug: Supportive Care",
    "Drug: Placebo",
]

CATEGORY_TO_COLUMN = {
    "Chemotherapy": "Drug: Chemotherapy",
    "Targeted Therapy": "Drug: Targeted Therapy",
    "Immunotherapy": "Drug: Immunotherapy / Biological Therapy",
    "Biological Therapy": "Drug: Immunotherapy / Biological Therapy",
    "Hormonal Therapy": "Drug: Hormonal Therapy",
    "Photodynamic Therapy": "Drug: Photodynamic Therapy",
    "Supportive Care": "Drug: Supportive Care",
    "Placebo": "Drug: Placebo",
}

# ============================================================
# 1. Load exclusion CSV
# ============================================================
print("Loading exclusion CSV...")
excl_df = pd.read_csv(CSV_PATH)

excl_df["rule"] = (
    excl_df["rule_text"]
    .astype(str)
    .str.replace("**", ": ", regex=False)
    .str.strip()
)

excl_df = excl_df[["cancer_type", "trial_id", "rule"] + EXCLUSION_COLS]

# ============================================================
# 2. Load rule summary JSON
# ============================================================
print("Loading rule summary JSON...")
with open(JSON_PATH, "r") as f:
    data = json.load(f)

rows = []

for cancer_type, cdata in data.items():
    for trial_id, tdata in cdata.get("trials", {}).items():

        for section in ["inclusion", "exclusion"]:
            for rule, rdata in tdata.get(section, {}).items():
                stats = rdata.get("rule_stats", {})

                rows.append({
                    "cancer_type": cancer_type,
                    "trial_id": trial_id,
                    "rule": rule,
                    "rule_section": section,

                    "overall_frequency": stats.get("overall_frequency"),
                    "n_trials": stats.get("n_trials"),
                    "enrollment_mean": stats.get("enrollment_mean"),
                    "enrollment_std": stats.get("enrollment_std"),
                    "site_mean": stats.get("site_mean"),
                    "site_std": stats.get("site_std"),
                    "recruitment_months_mean": stats.get("recruitment_months_mean"),
                    "recruitment_months_std": stats.get("recruitment_months_std"),
                    "epsm_mean": stats.get("epsm_mean"),
                    "epsm_std": stats.get("epsm_std"),
                    "start_date_mean": stats.get("start_date_mean"),
                    "start_date_std_yrs": stats.get("start_date_std_yrs"),
                })

rules_df = pd.DataFrame(rows)
print(f"Extracted {len(rules_df):,} rule instances")

# ============================================================
# 3. Merge exclusion rates
# ============================================================
print("Merging exclusion rates...")
merged_df = rules_df.merge(
    excl_df,
    on=["cancer_type", "trial_id", "rule"],
    how="left"
)

merged_df[EXCLUSION_COLS] = merged_df[EXCLUSION_COLS].fillna(0.0)

# ============================================================
# 4. Cancer-specific clustering (global unique IDs)
# ============================================================
print("Clustering rules per cancer...")
model = SentenceTransformer(MODEL_NAME)

global_cluster_id = 1
cluster_lookup = {}
cluster_center = {}

for cancer in merged_df["cancer_type"].unique():
    print(f"  → {cancer}")

    rules = (
        merged_df[merged_df["cancer_type"] == cancer]["rule"]
        .dropna()
        .unique()
        .tolist()
    )

    if len(rules) == 0:
        continue

    embeddings = model.encode(rules, show_progress_bar=True)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom"
    )

    labels = clusterer.fit_predict(embeddings)
    dfc = pd.DataFrame({"rule": rules, "label": labels})

    for lbl in sorted(dfc["label"].unique()):
        members = dfc[dfc.label == lbl]["rule"].tolist()

        if lbl == -1:
            # noise → singleton clusters
            for r in members:
                cid = global_cluster_id
                global_cluster_id += 1
                cluster_lookup[(cancer, r)] = cid
                cluster_center[cid] = r
        else:
            cid = global_cluster_id
            global_cluster_id += 1

            member_emb = model.encode(members, show_progress_bar=False)
            D = cosine_distances(member_emb)
            medoid_idx = D.mean(axis=1).argmin()
            center_rule = members[medoid_idx]

            for r in members:
                cluster_lookup[(cancer, r)] = cid

            cluster_center[cid] = center_rule

print(f"Total clusters created: {global_cluster_id - 1}")

# ============================================================
# 5. Attach cluster info
# ============================================================
merged_df["cluster_id"] = merged_df.apply(
    lambda r: cluster_lookup[(r["cancer_type"], r["rule"])],
    axis=1
)

merged_df["cluster_center_rule"] = merged_df["cluster_id"].map(cluster_center)

# ============================================================
# 6. Attach drug intervention categories (trial-level)
# ============================================================
print("Attaching drug intervention categories...")

with open(CLASSIFIED_JSON, "r") as f:
    classified = json.load(f)

trial_rows = []

for trial_id, record in classified.items():
    flags = {col: "No" for col in DRUG_COLUMNS}

    for item in record.get("interventions", []):
        cat = item.get("category")
        if cat in CATEGORY_TO_COLUMN:
            flags[CATEGORY_TO_COLUMN[cat]] = "Yes"

    trial_rows.append({
        "trial_id": trial_id,
        **flags
    })

drug_df = pd.DataFrame(trial_rows)

merged_df = merged_df.merge(
    drug_df,
    on="trial_id",
    how="left"
)

merged_df[DRUG_COLUMNS] = merged_df[DRUG_COLUMNS].fillna("No")

# ============================================================
# 7. Final sanity checks
# ============================================================
assert merged_df["cluster_id"].isna().sum() == 0
assert merged_df[EXCLUSION_COLS].isna().sum().sum() == 0

print("Rules with zero exclusion:",
      (merged_df[EXCLUSION_COLS].sum(axis=1) == 0).sum())

# ============================================================
# 8. Save
# ============================================================
merged_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved final CSV to {OUTPUT_PATH.resolve()}")
print("✅ Done.")