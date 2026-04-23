import faiss, torch, os, time, hdbscan, umap, joblib, hashlib, re
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import pandas as pd
import numpy as np
from pathlib import Path

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.decomposition import PCA
from plot import *
import json, collections

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()

# Setup OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Open log file
log_file = open("trial_design_log.txt", "w")

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

def json_serial(obj):
    """Fallback converter for JSON serialization."""
    import numpy as np, pandas as pd, datetime

    if isinstance(obj, (np.integer,)):               # ↪ Python int
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):               # ↪ list
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime.date, datetime.datetime)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def safe_epsm(row):
    try:
        enrollment = row["Enrollment Count"]
        site_count = row["Site Count"]
        recruitment_days = row["Recruitment Length (days)"]

        # Only compute if all values are valid and non-zero
        if pd.notna(enrollment) and pd.notna(site_count) and pd.notna(recruitment_days):
            if site_count > 0 and recruitment_days > 0:
                return enrollment / site_count / (recruitment_days / 30.44)
    except:
        pass
    return np.nan  

#gpt-4o-mini-2024-07-18 "gpt-4.1-2025-04-14"
def _gpt_relevance_check(rule, query, model) -> bool:
    prompt = f"""
You are an expert in clinical-trial eligibility criteria.

Determine whether the two eligibility rules describe the **same feature** 
and the **same numeric threshold**, ignoring whether the comparison sign 
is greater or less than (e.g., "<", ">", "≤", "≥").

If they mention the same biomarker or measurement (e.g., ALT, AST, Age, Platelet count)
and the same numeric value or multiplier (e.g., 1.5 × ULN, 100 × 10⁹/L, 18 years),
answer YES. Otherwise, answer NO.

Return exactly one word: YES or NO.

Eligibility Rule 1:
{query}

Eligibility Rule 2:
{rule}

Answer:
"""

    response = client.chat.completions.create(
        model= model,  # or "gpt-3.5-turbo"
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    result = response.choices[0].message.content.strip().lower()
    return result.startswith("y")

def log(message):
    print(message)
    log_file.write(str(message) + "\n")
    log_file.flush()

def safe_datetime_parse(date_str):
    try:
        return pd.to_datetime(date_str, errors="coerce")
    except:
        return pd.NaT

def days_between(start, end):
    if pd.isna(start) or pd.isna(end):
        return np.nan
    return (end - start).days

def encode_batch(model, tokenizer, texts, batch_size=64, max_length=512):
    if tokenizer is None:
        return model.encode(
        texts,
        batch_size=64,      
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False  
    )
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding in batches"):
        batch_texts = texts[i:i+batch_size]
        tokens = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        with torch.no_grad():
            outputs = model(**tokens.to(model.device))
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.extend(batch_embeddings)
    return np.array(embeddings)

def retrieve_similar_trial_ecs(model, tokenizer, user_query, trial_data, threshold):
    # ── Load & concatenate metadata ────────────────────────────────────────
    df = trial_data
    meta = (
        "Brief Title: "   + df["Brief Title"]      + "\n" +
        "Study Type: "    + df["Study Type"]       + "\n" +
        "Primary Purpose: "+ df["Primary Purpose"] + "\n" +
        "Keywords: "      + df["Keywords"]         + "\n" +
        "Interventions: " + df["Interventions"]    + "\n" +
        "Sex: "           + df["Sex"]              + "\n" +
        "Minimum Age: "   + df["Minimum Age"]      + "\n" +
        "Maximum Age: "   + df["Maximum Age"]
    ).tolist()

    if threshold == 0:
        return (
            df.iloc[:][["NCT ID", "Brief Title", "Eligibility Criteria"]]
                .to_dict(orient="records")
        )

    # ── Encode to embeddings (unit-norm) ───────────────────────────────────
    emb   = encode_batch(model, tokenizer, meta).astype("float32")
    q_emb = encode_batch(model, tokenizer, [user_query]).astype("float32")[0]

    faiss.normalize_L2(emb)
    faiss.normalize_L2(q_emb.reshape(1, -1))

    # ── threshold > 0 → FAISS range search ────────────────────────────────
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    lim, D, I = index.range_search(q_emb[np.newaxis, :], threshold)
    if lim[1] == 0:
        return []

    ids  = I[:lim[1]]
    sims = D[:lim[1]]

    order = np.argsort(-sims)
    ids   = ids[order]

    return (
        df.iloc[ids][["NCT ID", "Brief Title", "Eligibility Criteria"]]
          .to_dict(orient="records")
    )

def _load_or_make_ec_embeds(model, tokenizer, ec_texts: list[str], cache_dir: str = "../results/embeddings"):
    def _cache_key_for_rules(ec_texts: list[str]) -> str:
    # Hash the exact sequence of rule strings; changing order or content changes the key
        h = hashlib.sha1("\n".join(ec_texts).encode("utf-8")).hexdigest()
        return h[:100]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key_for_rules(ec_texts)
    cache_file = cache_dir / f"ec_embeds_{key}.npy"

    if cache_file.exists():
        ec_emb = np.load(cache_file, mmap_mode="r").astype("float32")  # mmap keeps RAM down
        # paranoid check
        if ec_emb.shape[0] == len(ec_texts):
            return ec_emb, str(cache_file)

    # (Re)compute and cache
    ec_emb = encode_batch(model, tokenizer, ec_texts).astype("float32")
    np.save(cache_file, ec_emb)
    return ec_emb, str(cache_file)

def retrieve_similar_ec(model, tokenizer, nli_model, nli_tokenizer, user_query, top_trials, threshold, evaluation = None):
    ec_rules = []                      # (rule, NCT, title)
    for t in top_trials:
        nct = t.get("NCT ID", "")
        ttl = t.get("Brief Title", "")
        for line in str(t.get("Eligibility Criteria", "")).splitlines():
            rule = line.strip()
            if rule:
                ec_rules.append((rule, nct, ttl))
    ec_texts = [r[0] for r in ec_rules]
    ec_texts = filter_numeric_text_rules(ec_texts)
    ec_rules = [(rule, nct, ttl) for (rule, nct, ttl) in ec_rules if rule in ec_texts]

    ec_emb, cache_path = _load_or_make_ec_embeds(model, tokenizer, ec_texts)

    ec_emb = ec_emb.astype("float32", copy=False)
    q_emb = encode_batch(model, tokenizer, [user_query]).astype("float32")[0]

    ec_tensor = torch.from_numpy(np.asarray(ec_emb)).to("cuda")   # or "cpu"
    q_tensor  = torch.from_numpy(q_emb).unsqueeze(0).to("cuda")

    sims = F.cosine_similarity(ec_tensor, q_tensor, dim=1)
    sims_np = sims.cpu().numpy()

    best_by_nct = {}
    for idx, score in enumerate(sims_np):
        rule, nct, title = ec_rules[idx]
        prev = best_by_nct.get(nct)
        if (prev is None) or (score > prev[1]):
            best_by_nct[nct] = (idx, float(score))

    winners = []
    nli_approved = 0
    llm_approved = 0
    for nct, (idx, score) in best_by_nct.items():
        if score < threshold:
            continue

        rule, _, title = ec_rules[idx]

        ok = True
        if evaluation == "llm":
            a1 = nli_relevance_check(rule, user_query, nli_model, nli_tokenizer)
            a2 = nli_relevance_check(user_query, rule, nli_model, nli_tokenizer)
            
            if ('neutral' in a1) or ('neutral' in a2):
                ok = False
            else:
                nli_approved += 1
                ok = _gpt_relevance_check(rule, user_query, "gpt-4o-mini-2024-07-18")
        elif evaluation == "nli":
            # Keep your current neutral-screen logic (cheap & robust)
            a1 = nli_relevance_check(rule, user_query, nli_model, nli_tokenizer)
            a2 = nli_relevance_check(user_query, rule, nli_model, nli_tokenizer)
            if ('neutral' in a1) or ('neutral' in a2):
                ok = False
        if ok:
            llm_approved += 1
            winners.append({
                "score": score,
                "rule":  rule,
                "nct_id": nct,
                "title":  title,
            })
    winners.sort(key=lambda x: x["score"])
    print(winners[:5])
    return winners

def nli_relevance_check(rule, user_query, model, tokenizer):
    text =  f"mednli: sentence1: {rule} sentence2: {user_query}"
    encoding = tokenizer.encode_plus(text, padding='max_length', max_length=256, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(model.device), encoding["attention_mask"].to(model.device)

    outputs = model.generate(input_ids=input_ids, attention_mask=attention_masks, max_length=8)
    for output in outputs:
        line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return line

def cluster_top_rules_and_rank_by_cluster(top_rules, topN_per_dataset=100):
    """
    For each dataset:
      1) take topN rules by frequency,
      2) embed with multi-qa-mpnet-base-cos-v1 (GPU, cached locally),
      3) reduce to 2D with UMAP,
      4) cluster with HDBSCAN (fallback: KMeans),
      5) compute average frequency per cluster,
      6) rank clusters and return representatives.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import umap, hdbscan
    from sklearn.cluster import KMeans

    model = SentenceTransformer(
        "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        cache_folder="/data/shelton/models",
        device="cuda"
    )

    out = {}
    for dataset, rules in top_rules.items():
        rules = sorted(rules, key=lambda r: r["overall_frequency"] or 0, reverse=True)[:topN_per_dataset]
        if not rules:
            out[dataset] = {"clusters": [], "rule_to_cluster": {}}
            continue

        names = [r["rule_name"] for r in rules]
        freqs = np.array([r["overall_frequency"] or 0.0 for r in rules], dtype=float)

        emb = model.encode(names, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)

        try:
            emb2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(emb)
            labels = hdbscan.HDBSCAN(min_cluster_size=5, metric="euclidean").fit_predict(emb2d)
        except Exception:
            n_clusters = min(20, max(2, len(rules)//5))
            labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(emb)

        clusters, rule_to_cluster = {}, {}
        for name, f, lab in zip(names, freqs, labels):
            rule_to_cluster[name] = int(lab)
            clusters.setdefault(int(lab), {"names": [], "freqs": []})
            clusters[int(lab)]["names"].append(name)
            clusters[int(lab)]["freqs"].append(f)

        ranked = []
        for lab, info in clusters.items():
            arr = np.array(info["freqs"], dtype=float)
            avgf = float(arr.mean()) if len(arr) else 0.0
            idx = int(np.argmax(arr)) if len(arr) else 0
            rep_name = info["names"][idx] if info["names"] else None
            rep_freq = float(arr[idx]) if len(arr) else 0.0
            ranked.append({
                "cluster_id": lab,
                "avg_frequency": avgf,
                "n_rules": len(info["names"]),
                "representative_rule": rep_name,
                "representative_freq": rep_freq,
                "rule_names": info["names"],
            })
        ranked.sort(key=lambda c: (c["avg_frequency"], c["representative_freq"]), reverse=True)
        out[dataset] = {"clusters": ranked, "rule_to_cluster": rule_to_cluster}
    return out

def select_top_unique_rules_with_gpt(top_rules, k=10):
    """
    For each dataset (rules sorted by frequency):
      1) keep the first rule,
      2) for each next rule, ask GPT if it is the SAME as any already kept,
      3) include only if GPT says NOT the same,
      4) stop when k rules are kept.
    """
    import os, re, json
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def is_same_rule(a, b):
        prompt = (
            "You are comparing two clinical trial eligibility rules. "
            "Answer with a single word: YES if they are the SAME criterion (paraphrases), "
            "NO if they are meaningfully DIFFERENT.\n\n"
            f"Rule A:\n{a}\n\nRule B:\n{b}\n\nAnswer:"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        ans = resp.choices[0].message.content.strip().upper()
        return ans.startswith("Y")

    out = {}
    for dataset, rules in top_rules.items():
        kept, kept_texts = [], []
        for r in rules:
            if len(kept) >= k: break
            candidate = r["rule_name"]
            if any(candidate.lower() == t.lower() for t in kept_texts): continue

            dup = False
            for t in kept_texts:
                try:
                    if is_same_rule(candidate, t):
                        dup = True; break
                except Exception:
                    dup = False
            if not dup:
                kept.append(r)
                kept_texts.append(candidate)
        out[dataset] = kept
    return out

def filter_numeric_text_rules(rules, keywords=NUMERIC_KEYWORDS):
    """
    Filter list of strings (like 'feature: value') keeping only those
    that contain numeric-related terms or comparison symbols.
    """
    keyword_pattern = re.compile(
        r"\b(" + "|".join([re.escape(k) for k in keywords]) + r")\b",
        re.IGNORECASE
    )
    operator_pattern = re.compile(r"(<=|>=|<|>|=|≤|≥|\d+)")  # also keep explicit numbers

    filtered = []
    for rule in rules:
        if keyword_pattern.search(rule) or operator_pattern.search(rule):
            filtered.append(rule)
    return filtered

def main():
    start = time.time()
    # model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1", cache_dir="/data/shelton/models").to("cuda")
    # tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1", cache_dir="/data/shelton/models")

    model = SentenceTransformer(
    "sentence-transformers/multi-qa-mpnet-base-cos-v1",
    cache_folder="/data/shelton/models",
    device="cuda"   # or "cuda:0"
    )

    tokenizer = None

    nli_tokenizer = AutoTokenizer.from_pretrained("razent/SciFive-large-Pubmed_PMC-MedNLI", cache_dir = "/data/shelton/models")
    nli_model = AutoModelForSeq2SeqLM.from_pretrained("razent/SciFive-large-Pubmed_PMC-MedNLI", cache_dir = "/data/shelton/models", device_map="auto")

    # applications = ["breast_cancer", "colorectal_cancer", "lung_cancer", "melanoma", "prostate_cancer"]
    applications = [ "prostate_cancer"]

    results = collections.defaultdict(
        lambda: {
            "general_stats": {},
            "trials": collections.defaultdict(lambda: {"inclusion": collections.defaultdict(dict), "exclusion": collections.defaultdict(dict)})
        }
    )

    for application in applications:
        trial_csv = f"../results/retrieved_trials_{application}.csv"
        trial_data = pd.read_csv(trial_csv).fillna("")

        top_trials = retrieve_similar_trial_ecs(
            model=model,
            tokenizer=tokenizer,
            user_query="N/A",
            trial_data=trial_data,
            threshold=0.0,
        )

        results[application]["general_stats"]["n_trials_total"] = len(trial_data)
    
        base = "/home/konghaoz/activeLearning_trialRecruit/data/application"
        inclusion_trial_data = pd.read_csv(f"{base}/{application}/structured_trial_data_gpt-4.1-2025-04-14_inclusion_application.csv")
        exclusion_trial_data = pd.read_csv(f"{base}/{application}/structured_trial_data_gpt-4.1-2025-04-14_exclusion_application.csv")

        for trial_id in inclusion_trial_data["Trial_ID"]:
            
            incl_df = (inclusion_trial_data[inclusion_trial_data["Trial_ID"] == trial_id].dropna(axis=1, how="all").drop("Trial_ID", axis=1))
            incl_df.columns = [col.split(": ")[1].lower() if ": " in col else col.lower()for col in incl_df.columns]
            inclusion_criteria = [f"{feat}: {val}" for feat, val in zip(incl_df.columns, incl_df.iloc[0])]

            excl_df = (exclusion_trial_data[exclusion_trial_data["Trial_ID"] == trial_id].dropna(axis=1, how="all").drop("Trial_ID", axis=1))
            excl_df.columns = [col.split(": ")[1].lower() if ": " in col else col.lower()for col in excl_df.columns]
            exclusion_criteria = [f"{feat}: {val}" for feat, val in zip(excl_df.columns, excl_df.iloc[0])]

            inclusion_criteria = filter_numeric_text_rules(inclusion_criteria)
            exclusion_criteria = filter_numeric_text_rules(exclusion_criteria)

            for idx, criterion in enumerate(inclusion_criteria + exclusion_criteria):
                print(f"\n{application} (total trials: {len(top_trials)}) {trial_id}\n{criterion}\n")
                bucket = "inclusion" if idx < len(inclusion_criteria) else "exclusion"

                top_ec = retrieve_similar_ec(
                    model=model,
                    tokenizer=tokenizer,
                    nli_model=nli_model,
                    nli_tokenizer = nli_tokenizer,
                    user_query=criterion,
                    top_trials=top_trials,
                    threshold=0.60,
                    evaluation="llm"
                )

                unique_ids = {ec["nct_id"] for ec in top_ec}
                trial_selected = trial_data[trial_data["NCT ID"].isin(unique_ids)].copy()
                if trial_selected.empty:
                    continue

                trial_selected["Start Date"] = trial_selected["Start Date"].apply(safe_datetime_parse)
                trial_selected["Primary Completion Date"] = trial_selected["Primary Completion Date"].apply(safe_datetime_parse)
                trial_selected["Recruitment Length (days)"] = trial_selected.apply(lambda r: days_between(r["Start Date"], r["Primary Completion Date"]),axis=1)
                trial_selected["Site Count"] = (trial_selected["Locations"].fillna("").apply(lambda s: 0 if s.strip() == "" else s.count(";") + 1))
                trial_selected["EPSM"] = trial_selected.apply(safe_epsm, axis=1)

                trial_selected["Enrollment Count"] = pd.to_numeric(trial_selected["Enrollment Count"], errors="coerce")
                trial_selected["Site Count"] = pd.to_numeric(trial_selected["Site Count"], errors="coerce")
                trial_selected["Recruitment Length (days)"] = pd.to_numeric(trial_selected["Recruitment Length (days)"], errors="coerce")
                trial_selected["EPSM"] = pd.to_numeric(trial_selected["EPSM"], errors="coerce")

                enrollment = trial_selected["Enrollment Count"]
                sites = trial_selected["Site Count"]
                months = trial_selected["Recruitment Length (days)"] / 30.44
                epsm_valid = trial_selected["EPSM"].dropna()
                start_dates = trial_selected["Start Date"].dropna().values.astype(
                    "datetime64[D]"
                )

                rule_stats = {
                    "overall_frequency": len(trial_selected) / len(trial_data),
                    "n_trials": len(trial_selected),
                    "enrollment_mean": enrollment.mean(),
                    "enrollment_std": enrollment.std(),
                    "site_mean": sites.mean(),
                    "site_std": sites.std(),
                    "recruitment_months_mean": months.mean(),
                    "recruitment_months_std": months.std(),
                    "epsm_mean": epsm_valid.mean(),
                    "epsm_std": epsm_valid.std(),
                    "start_date_mean": None,
                    "start_date_std_yrs": None,
                }

                if len(start_dates):
                    ts = start_dates.astype("datetime64[s]").astype(float)
                    mean_ts, std_ts = ts.mean(), ts.std()
                    rule_stats["start_date_mean"] = str(
                        datetime.utcfromtimestamp(mean_ts).date()
                    )
                    rule_stats["start_date_std_yrs"] = round((std_ts / 86400) / 365.25, 2)

                rule_slot = results[application]["trials"][trial_id][bucket][criterion]
                rule_slot["rule_stats"] = rule_stats
                rule_slot["matched_trials"] = []

                for tr in top_ec:
                    row_df = trial_selected.loc[trial_selected["NCT ID"] == tr["nct_id"]]                   
                    if row_df.empty:
                        continue
                    r = row_df.iloc[0]
                    rule_slot["matched_trials"].append(
                        {
                            "nct_id": tr["nct_id"],
                            "rule": tr["rule"],
                            "trial_stats": {
                                "enrollment": r["Enrollment Count"],
                                "site_count": r["Site Count"],
                                "start_date": str(r["Start Date"].date()) if pd.notna(r["Start Date"]) else None,
                                "completion_date": str(r["Primary Completion Date"].date()) if pd.notna(r["Primary Completion Date"]) else None,
                                "recruitment_days": int(r["Recruitment Length (days)"]) if pd.notna(r["Recruitment Length (days)"]) else None,
                                "epsm": r["EPSM"],
                            }
                        }
                    )

        cache = Path("../results/ec_temporary_embedding.npy")
        if cache.exists():
            cache.unlink()

    out_path = Path(f"../results/full_rule_summary_nli_check_0.6_numerical_{application}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=json_serial)
    print(f"Finish in {(time.time()-start)/3600} hrs")

if __name__ == "__main__":
    main()