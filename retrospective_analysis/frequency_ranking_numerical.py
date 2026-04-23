import json,os,re,umap,hdbscan,matplotlib
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from collections import defaultdict
from sentence_transformers import SentenceTransformer

import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt


matplotlib.rcParams['svg.fonttype'] = 'none'

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

NUMERIC_KEYWORDS = [
    "less than", "greater than", "equal to", "at least",
    # Hematology
    "white blood cell", "wbc", "leukocyte",
    "platelet", "plt",
    "hemoglobin", "hgb",
    "hematocrit", "hct",
    "red blood cell", "rbc",
    "absolute neutrophil count", "anc", "neutrophil",
    "lymphocyte", "lymphocytes",

    # Liver function
    "aspartate aminotransferase", "ast", "sgot",
    "alanine aminotransferase", "alt", "sgpt",
    "bilirubin", "total bilirubin", "direct bilirubin",
    "alkaline phosphatase", "alp",
    "albumin", "total protein",

    # Renal function
    "creatinine", "serum creatinine",
    "creatinine clearance", "crcl",
    "estimated glomerular filtration rate", "egfr",
    "blood urea nitrogen", "bun", "urea",

    # Electrolytes & minerals
    "sodium", "na",
    "potassium", "k",
    "calcium", "ca",
    "magnesium", "mg",
    "phosphate", "phosphorus", "po4",

    # Coagulation
    "prothrombin time", "pt",
    "international normalized ratio", "inr",
    "activated partial thromboplastin time", "aptt", "ptt",

    # Metabolic / others
    "lactate dehydrogenase", "ldh",
    "uric acid",
    "glucose", "fasting glucose", "blood sugar",
    "cholesterol", "hdl", "ldl",
    "triglyceride",
    "oxygen saturation", "spo2"
]



def build_prompt(rules_batch):
    numbered = "\n".join([f"{i+1}. {r['rule_name']}" for i, r in enumerate(rules_batch)])
    return f"""
You are an expert in clinical trial eligibility criteria.

Task: Normalize each rule into a JSON object. 

Guidelines:
- If the rule mentions exactly ONE numerical feature, normalize to:
  {{ "<Canonical Feature Name>": "<Operator + Threshold>" }}
  Examples:
    - "min hemoglobin level (g/dl): 9.0" : {{ "Hemoglobin": ">= 9.0 g/dl" }}
    - "serum creatinine < 1.5 × ULN" : {{ "Creatinine": "< 1.5 × ULN" }}
- If the rule mentions MULTIPLE distinct numerical conditions (e.g., "AST ≤ 3 × ULN, or ≤ 5 × ULN if liver metastases present"):
  return {{ "the original text": "multiple" }}
- For all other rules (non-numerical, descriptive, or categorical):
  return {{ "the original text": "N/A" }}

Canonicalization rules:
- ANC: Absolute neutrophil count  
- WBC: White blood cell count  
- Hb / Hgb: Hemoglobin  
- Plt: Platelet count  
- ALT: Alanine aminotransferase  
- AST: Aspartate aminotransferase  
- INR: International normalized ratio  
- APTT: Activated partial thromboplastin time  
- ULN: Upper limit of normal (always keep "ULN" in the string, do not expand)

Formatting rules:
- Always use singular canonical form (e.g., "Platelet count", not "Platelets").  
- Preserve the SAME sequence as input order.  
- Return ONLY a JSON array (list), no explanation, no extra text.  

Rules to normalize:
{numbered}
"""

def call_model(prompt, model="gpt-4o"):
    """Send prompt to model and parse JSON output."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = resp.choices[0].message.content.strip()

    # Remove ``` fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("Not a list")
        return parsed
    except Exception:
        print("⚠️ JSON parse failed, here’s the raw output head:\n", raw[:400])
        exit()

# --- Utility: normalize thresholds to collapse equivalents ---
def normalize_threshold(value: str) -> str:
    if value in ("N/A", "multiple", None):
        return None

    v = value.strip().lower()

    # Normalize common patterns
    v = v.replace("cells/mm3", "/µl")
    v = v.replace("cells/µl", "/µl")
    v = v.replace("mm3", "/µl")
    v = v.replace("x 10^9/l", "×10^9/L")
    v = v.replace("10^9/l", "×10^9/L")
    v = v.replace("≥", ">=")
    v = v.replace("≤", "<=")

    # Clean multiple spaces
    v = re.sub(r"\s+", " ", v).strip()
    return v

def filter_numeric_rules(results, keywords=NUMERIC_KEYWORDS):
    """
    Return rules whose rule_name mentions:
      - Any numeric lab keyword (full name or abbreviation), OR
      - Any comparison operator (<, >, ≤, ≥, =) suggesting numeric threshold.
    """
    keyword_pattern = re.compile(r"\b(" + "|".join([re.escape(k) for k in keywords]) + r")\b",
                                 re.IGNORECASE)
    operator_pattern = re.compile(r"(<=|>=|<|>|=|≤|≥)")  # matches <, >, <=, >=, =

    filtered = []
    for r in results:
        name = r["rule_name"]
        if keyword_pattern.search(name) or operator_pattern.search(name):
            filtered.append(r)
    return filtered


def summarize_thresholds(merged_results, output_file):
    """
    Organize thresholds per dataset → feature → threshold,
    but only store the first rule that introduced the threshold.
    """
    summary = defaultdict(lambda: defaultdict(dict))

    for r in merged_results:
        dataset = r["dataset"]
        feature = r["normalized_name"]
        value = r["normalized_value"]

        # Skip invalid values
        if value in ("N/A", "multiple", None):
            continue

        # Only add the first occurrence of this threshold
        if value not in summary[dataset][feature]:
            summary[dataset][feature][value] = {
                "trial_id": r["trial_id"],
                "rule_type": r["rule_type"],
                "rule_name": r["rule_name"],
                "rule_stats": r.get("rule_stats", {}) or {}
            }

    # Convert defaultdict to dict
    final = {ds: dict(feats) for ds, feats in summary.items()}

    with open(output_file, "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved first-rule threshold summary → {output_file}")
    return final


def main():
    applications = ["breast_cancer","colorectal_cancer","lung_cancer", "melanoma", "prostate_cancer"]
    for application in applications:
        OUTPUT_FILE = f"/home/konghaoz/trialDesigner/results/numerical_normalization_{application}.json"
        BATCH_SIZE = 100

        json_file = f"/home/konghaoz/trialDesigner/results/full_rule_summary_nli_check_0.6_numerical_{application}.json"
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

        results = filter_numeric_rules(results)

        # normalized = []
        # for i in tqdm(range(0, len(results), BATCH_SIZE)):
        #     batch = results[i:i+BATCH_SIZE]
        #     prompt = build_prompt(batch)
        #     out = call_model(prompt)

        #     for r, norm in zip(batch, out):
        #         normalized.append({
        #             "trial_id": r["trial_id"],
        #             "dataset": r["dataset"],
        #             "rule_type": r["rule_type"],
        #             "original_rule": r["rule_name"],
        #             "normalized_name": list(norm.keys())[0],
        #             "normalized_value": list(norm.values())[0]
        #         })

        # with open(OUTPUT_FILE, "w") as f:
        #     json.dump(normalized, f, indent=2, ensure_ascii=False)

        # print(f"✅ Saved {len(normalized)} normalized rules to {OUTPUT_FILE}")

        with open(OUTPUT_FILE, "r") as f:
            normalized_data = json.load(f)

        merged_results = []
        for raw, norm in zip(results, normalized_data):
            raw["normalized_name"] = norm["normalized_name"]
            raw["normalized_value"] = norm["normalized_value"]
            merged_results.append(raw)

        summary = summarize_thresholds(
            merged_results,
            f"/home/konghaoz/trialDesigner/results/numerical_threshold_summary_{application}.json"
        )

if __name__ == "__main__":
    main()