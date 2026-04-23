import requests, re, argparse
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Iterable, Tuple, Union


# ── Helper --------------------------------------------------------------
def safe_epsm(row: pd.Series):
    """
    Enrollment-per-site-per-month with robust NaN handling.
    """
    e   = row["Enrollment Count"]
    s   = row["Site Count"]
    dur = row["Recruitment Length (days)"]

    if pd.isna(e) or pd.isna(s) or pd.isna(dur) or s <= 0 or dur <= 0:
        return np.nan
    return e / s / (dur / 30.44)


# ── Summary -------------------------------------------------------------
def summarize_trial_stats(df: pd.DataFrame) -> None:
    """
    Prints corpus-level summary statistics, using the same EPSM recipe
    you use elsewhere (site-count parsing, day-level duration, and the
    `safe_epsm` guard).
    """
    df = df.copy()

    # ---- Parse numbers & dates ----------------------------------------
    df["Enrollment Count"] = pd.to_numeric(df["Enrollment Count"], errors="coerce")

    df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce", utc=True)
    df["Primary Completion Date"] = pd.to_datetime(
        df["Primary Completion Date"], errors="coerce", utc=True
    )

    # ---- Compute recruitment length (days) ----------------------------
    df["Recruitment Length (days)"] = (
        df["Primary Completion Date"] - df["Start Date"]
    ).dt.days

    # ---- Site count: same rule you use in safe_epsm -------------------
    df["Site Count"] = (
        df["Locations"]
        .fillna("")
        .apply(lambda s: 0 if s.strip() == "" else s.count(";") + 1)
    )

    # ---- EPSM with the exact same guard -------------------------------
    df["EPSM"] = df.apply(safe_epsm, axis=1)

    # ---- Keep rows with all required fields ---------------------------
    clean = df.dropna(
        subset=[
            "Enrollment Count",
            "Recruitment Length (days)",
            "Site Count",
            "EPSM",
            "Start Date",
        ]
    )

    # ---- Average start date ± std -------------------------------------
    ts = clean["Start Date"].map(lambda x: x.to_pydatetime().timestamp())
    mean_ts = ts.mean()
    std_ts  = ts.std()

    mean_start = datetime.utcfromtimestamp(mean_ts).date()
    std_years  = round((std_ts / 86400) / 365.25, 2)

    # ---- Print summary -------------------------------------------------
    print("📊 Trial Summary Statistics")
    print(f"Number of trials: {len(df)}")
    print(
        f"Average enrollment: "
        f"{clean['Enrollment Count'].mean():.1f} ± {clean['Enrollment Count'].std():.1f}"
    )
    print(
        f"Average recruitment length: "
        f"{(clean['Recruitment Length (days)'] / 30.44).mean():.2f} "
        f"± {(clean['Recruitment Length (days)'] / 30.44).std():.2f} months"
    )
    print(
        f"Average site count: "
        f"{clean['Site Count'].mean():.2f} ± {clean['Site Count'].std():.2f}"
    )
    print(
        f"Average enrollment per site per month (EPSM): "
        f"{clean['EPSM'].mean():.2f} ± {clean['EPSM'].std():.2f}"
    )
    print(f"Average start date: {mean_start} ± {std_years} years")

# def summarize_trial_stats(df: pd.DataFrame) -> None:
#     df = df.copy()

#     # Convert and clean relevant columns
#     df["Enrollment Count"] = pd.to_numeric(df["Enrollment Count"], errors="coerce")
#     df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce", utc=True)
#     df["Primary Completion Date"] = pd.to_datetime(df["Primary Completion Date"], errors="coerce", utc=True)

#     # Drop rows with missing critical values
#     clean_df = df.dropna(subset=["Enrollment Count", "Start Date", "Primary Completion Date"])

#     # Recruitment duration in months
#     clean_df["Recruitment Length (Months)"] = (
#         (clean_df["Primary Completion Date"] - clean_df["Start Date"]).dt.days / 30.44
#     )

#     # Estimate number of sites
#     clean_df["Site Count"] = df["Locations"].apply(lambda x: len(str(x).split(");")) if pd.notna(x) else 0)

#     # Enrollment per site per month
#     clean_df["Enrollment per Site per Month"] = clean_df["Enrollment Count"] / (
#         clean_df["Site Count"] * clean_df["Recruitment Length (Months)"]
#     )

#     # Compute average start date ± std
#     start_dates = clean_df["Start Date"].dropna().map(lambda x: x.to_pydatetime().timestamp())
#     mean_ts = np.mean(start_dates)
#     std_ts = np.std(start_dates)
#     mean_start = datetime.utcfromtimestamp(mean_ts).date()
#     std_days = std_ts / 86400
#     std_years = round(std_days / 365.25, 2)

#     # Print summary
#     print("📊 Trial Summary Statistics")
#     print(f"Number of trials: {len(df)}")
#     print(f"Average enrollment: {clean_df['Enrollment Count'].mean():.1f} ± {clean_df['Enrollment Count'].std():.1f}")
#     print(f"Average recruitment length: {clean_df['Recruitment Length (Months)'].mean():.2f} ± {clean_df['Recruitment Length (Months)'].std():.2f} months")
#     print(f"Average site count: {clean_df['Site Count'].mean():.2f} ± {clean_df['Site Count'].std():.2f}")
#     print(f"Average enrollment per site per month: {clean_df['Enrollment per Site per Month'].mean():.2f} ± {clean_df['Enrollment per Site per Month'].std():.2f}")
#     print(f"Average start date: {mean_start} ± {std_years} years")
    
def recursive_dump(obj: Any,
                   file: Union[str, Path] = "structure.txt",
                   *,
                   prefix: str = "",
                   max_depth: int | None = None,
                   max_items: int | None = None,
                   encoding: str = "utf-8") -> Path:
    """
    Walk a nested Python structure (dict / list / tuple / set / scalar) and
    write a nicely‑indented outline to `file`.

    Returns
    -------
    Path
        The path to the file written.
    """
    # --- internal pretty‑printer --------------------------------------------
    lines: list[str] = []

    def _recurse(node: Any, indent: int = 0) -> None:
        spacer = "  " * indent

        # depth limit
        if max_depth is not None and indent >= max_depth:
            lines.append(f"{spacer}{prefix}…")
            return

        # scalar
        if not isinstance(node, (dict, list, tuple, set)):
            lines.append(f"{spacer}{prefix}{node!r}")
            return

        # iterable (dict or sequence)
        def _items(x: Any) -> Iterable[Tuple[Any, Any]]:
            return x.items() if isinstance(x, dict) else enumerate(x)

        for i, (k, v) in enumerate(_items(node)):
            if max_items is not None and i >= max_items:
                lines.append(f"{spacer}  …")
                break
            label = f"{k}:" if isinstance(node, dict) else f"[{k}]"
            lines.append(f"{spacer}{prefix}{label}")
            _recurse(v, indent + 1)

    _recurse(obj)

    # --- write to disk -------------------------------------------------------
    out_path = Path(file)
    out_path.write_text("\n".join(lines), encoding=encoding)

# def split_inclusion_exclusion(text):
#     """
#     Split the eligibility textblock into inclusion and exclusion strings.
#     Handles variants like 'Key inclusion criteria', extra spaces, and line breaks.
#     """
#     # Use regex, tolerant to variants and line breaks
#     match = re.search(
#         r"(Inclusion Criteria|Key inclusion criteria|Inclusion)\s*[:\-]?\s*(.*?)\s*(Exclusion Criteria|Key exclusion criteria|Exclusion)\s*[:\-]?\s*(.*)",
#         text,
#         re.DOTALL | re.IGNORECASE
#     )
#     if match:
#         inclusion = match.group(2).strip()
#         exclusion = match.group(4).strip()
#     else:
#         print(text)
#         exit()
#         inclusion = text.strip()
#         exclusion = ""
    
#     return inclusion, exclusion

def fetch_by_nct(nct_id):
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

manual_ids_map = {
    "lung cancer": ["NCT01483144", "NCT02270645", "NCT03401229", "NCT03924869"],
    "melanoma": ["NCT02967692"],
}

parser = argparse.ArgumentParser(description="Retrieve clinical trials from ClinicalTrials.gov.")
parser.add_argument("-query", type=str, default="cancer", help="Query condition string (e.g., 'Cancer OR Tumor')")
parser.add_argument("-size", type=int, default=-1, help="Max number of trials to retrieve (-1 = all)")
args = parser.parse_args()

base_url = "https://clinicaltrials.gov/api/v2/studies"

params = {
    "query.cond": args.query,
}

#python3 fetch_clinicalTrial.py -query "breast cancer" -size -1 && python3 fetch_clinicalTrial.py -query "colorectal cancer" -size -1 && python3 fetch_clinicalTrial.py -query "lung cancer" -size -1 && python3 fetch_clinicalTrial.py -query "melanoma" -size -1 && python3 fetch_clinicalTrial.py -query "prostate cancer" -size -1



data_list = []

trialSize = 0
while True:
    print("Fetching:", base_url + '?' + '&'.join([f"{k}={v}" for k, v in params.items()]))
    if args.size > 0:
        remaining = args.size - len(data_list)
        if remaining <= 0:
            break
        params["pageSize"] = min(remaining, 1000)
    else:
        params["pageSize"] = 1000  # default max    

    resp = requests.get(base_url, params=params)
    resp.raise_for_status()

    payload = resp.json()
    studies = payload.get("studies", [])
    for study in studies:
        recursive_dump(study,"log.txt",)
        p = study.get("protocolSection", {})

        idm = p.get("identificationModule", {})
        sm = p.get("statusModule", {})
        cm = p.get("conditionsModule", {})
        dm = p.get("designModule", {})
        aim = p.get("armsInterventionsModule", {})
        elm = p.get("eligibilityModule", {})
        locm = p.get("contactsLocationsModule", {})
        desc = p.get("descriptionModule", {})


        overallStatus = sm.get("overallStatus")
        phase = str(dm.get("phases", []))
        if (not overallStatus == "COMPLETED") or (not "PHASE3" in phase):
            continue

        record = {
            # Identification
            "NCT ID": idm.get("nctId"),
            "Org Study ID": idm.get("orgStudyIdInfo", {}).get("id"),
            "Organization Name": idm.get("organization", {}).get("fullName"),
            "Brief Title": idm.get("briefTitle"),
            "Official Title": idm.get("officialTitle"),
            "Acronym": idm.get("acronym"),

            # Status
            "Overall Status": sm.get("overallStatus"),
            "Start Date": sm.get("startDateStruct", {}).get("date"),
            "Primary Completion Date": sm.get("primaryCompletionDateStruct", {}).get("date"),
            "Completion Date": sm.get("completionDateStruct", {}).get("date"),
            "Study First Submit Date": sm.get("studyFirstSubmitDate"),
            "Study First Post Date": sm.get("studyFirstPostDateStruct", {}).get("date"),
            "Last Update Post Date": sm.get("lastUpdatePostDateStruct", {}).get("date"),

            # Design
            "Study Type": dm.get("studyType"),
            "Phases": ', '.join(dm.get("phases", [])),
            "Intervention Model": dm.get("designInfo", {}).get("interventionModel"),
            "Primary Purpose": dm.get("designInfo", {}).get("primaryPurpose"),
            "Masking": dm.get("designInfo", {}).get("maskingInfo", {}).get("masking"),
            "Enrollment Count": dm.get("enrollmentInfo", {}).get("count"),

            # Conditions & Keywords
            "Conditions": ', '.join(cm.get("conditions", [])),
            "Keywords": ', '.join(cm.get("keywords", [])),

            # Arms & Interventions
            "Arms": '; '.join([arm.get("label", "") for arm in aim.get("armGroups", [])]),
            "Interventions": '; '.join([
                f"{i.get('type', '')}: {i.get('name', '')} ({i.get('description', '')})"
                for i in aim.get("interventions", [])
            ]),

            # Outcomes
            "Primary Outcomes": '; '.join([
                f"{o.get('measure', '')}: {o.get('description', '')} ({o.get('timeFrame', '')})"
                for o in p.get("outcomesModule", {}).get("primaryOutcomes", [])
            ]),
            "Secondary Outcomes": '; '.join([
                f"{o.get('measure', '')}: {o.get('description', '')} ({o.get('timeFrame', '')})"
                for o in p.get("outcomesModule", {}).get("secondaryOutcomes", [])
            ]),

            # Eligibility
            "Eligibility Criteria": elm.get("eligibilityCriteria"),
            "Healthy Volunteers": elm.get("healthyVolunteers"),
            "Sex": elm.get("sex"),
            "Minimum Age": elm.get("minimumAge"),
            "Maximum Age": elm.get("maximumAge"),
            "Standard Ages": ', '.join(elm.get("stdAges", [])),

            # Locations
            "Locations": '; '.join([
                f"{loc.get('facility', '')} ({loc.get('city', '')}, {loc.get('state', '')}), {loc.get('country', '')}"
                for loc in locm.get("locations", [])
            ]),

            # Description Module (Full)
            "Brief Summary": desc.get("briefSummary"),
            "Detailed Description": desc.get("detailedDescription"),
        }

        data_list.append(record)

    token = payload.get("nextPageToken")
    if token:
        params["pageToken"] = token
    else:
        break

print(len(data_list))
data = pd.DataFrame(data_list)
data.to_csv(f"../results/retrieved_trials_{args.query}.csv")
plot_basic_info(data, fname="../results/basic_stats.png")
summarize_trial_stats(data)