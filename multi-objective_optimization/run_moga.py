# +
import os, multiprocessing, argparse, random, torch, re, json
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from moga import MOGA

def read_json_data(application_path):
    merged_data = {}
    for filename in sorted(os.listdir(application_path)):
        if filename.endswith("_results.json"):
            file_path = os.path.join(application_path, filename)

            with open(file_path, "r") as f:
                content = f.read()
            fixed_content = re.sub(r'}\s*{(?=\s*"NCT\d{8}")', r',\n', content)
            try:
                file_data = json.loads(fixed_content)

                # Merge each trial and its patients
                for trial_id, patients in file_data.items():
                    if trial_id not in merged_data:
                        merged_data[trial_id] = {}
                    for patient_id, patient_data in patients.items():
                        if patient_id in merged_data[trial_id]:
                            print(f"Duplicate patient {patient_id} in trial {trial_id}, skipping...")
                        else:
                            merged_data[trial_id][patient_id] = patient_data
                            
            except json.JSONDecodeError as e:
                print(f"\n JSONDecodeError: {e}")
                print(f"⛳ Error at line {e.lineno}, column {e.colno}, char position {e.pos}")
                    # Show error context
                lines = fixed_content.splitlines()
                error_line = e.lineno - 1
                print("\n🔍 Context around the error:\n")
                for i in range(max(0, error_line - 3), min(len(lines), error_line + 3)):
                    prefix = ">>> " if i == error_line else "    "
                    print(f"{prefix}Line {i+1}: {lines[i]}")
    return merged_data

def extract_eligibility(trial, path):
    inclusion_trial_data = pd.read_csv(os.path.join(path, "structured_trial_data_gpt-4.1-2025-04-14_inclusion_application.csv"))
    exclusion_trial_data = pd.read_csv(os.path.join(path, "structured_trial_data_gpt-4.1-2025-04-14_exclusion_application.csv"))

    cur_trial_inclusion = inclusion_trial_data[inclusion_trial_data["Trial_ID"] == trial].dropna(axis=1, how='all')
    cur_trial_inclusion = cur_trial_inclusion.drop("Trial_ID", axis=1)
    cur_trial_inclusion.columns = [col.split(": ")[1].lower() if ": " in col else col.lower() for col in cur_trial_inclusion.columns]

    cur_trial_exclusion = exclusion_trial_data[exclusion_trial_data["Trial_ID"] == trial].dropna(axis=1, how='all')
    cur_trial_exclusion = cur_trial_exclusion.drop("Trial_ID", axis=1)
    cur_trial_exclusion.columns = [col.split(": ")[1].lower() if ": " in col else col.lower() for col in cur_trial_exclusion.columns]
    
    inclusion_criteria = []
    for _, row in cur_trial_inclusion.iterrows():
            for col, val in row.items():
                if val == "descriptive":
                    rule_str = f"{col}"
                else:
                    rule_str = f"{col}: {val}"
                inclusion_criteria.append(rule_str)
    
    exclusion_criteria = []
    for _, row in cur_trial_exclusion.iterrows():
        for col, val in row.items():
            if val == "descriptive":
                rule_str = f"{col}"
            else:
                rule_str = f"{col}: {val}"
            exclusion_criteria.append(rule_str)
    
    return inclusion_criteria, exclusion_criteria
    
def main(args, working_dir):
    trial = "NCT02566993"
    merged_json_path = os.path.join(working_dir, "results", args.dataset, "merged_lung_cancer_0.5_nli_NA.json")
    if os.path.exists(merged_json_path):
        with open(merged_json_path, "r") as f:
            results = json.load(f)
#     else:
#         results = read_json_data(os.path.join(working_dir, "results", args.dataset))
#         with open(merged_json_path, "w") as f:
#             json.dump(results, f, indent=2)
        
    patient_data = pd.read_csv(os.path.join(working_dir, "data", args.dataset, "structured_patient_data.csv"))
    patient_data = patient_data[["patient_id", "demographic: gender", "demographic: ethnicity"]]
    patient_data.columns = ["patient_id", "gender", "ethnicity"]
    inc_full, exc_full = extract_eligibility(trial, path = "../data/lung_cancer/")
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    model = AutoModelForSequenceClassification.from_pretrained("./adverse_event").to("cuda")
    
    
#     for ft in ["enrollment+adverse", "enrollment+size", "enrollment+size+adverse"]:
    for ft in ["enrollment+adverse"]:
        for trial_id, patients in results.items():
            if trial_id == trial:
                moga = MOGA(patients, patient_data, "ethnicity", os.path.join(working_dir, "results" , args.dataset, ft), inc_full, exc_full, model, tokenizer, numGen = args.numGen, sizePop = args.popSize, seed=args.seed, min_active_rules=args.min_active_rules,fitness_type=ft)
                best_solution = moga.fit_transform(X=None)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default = "lung_cancer")
    parser.add_argument('-numGen', type=int, default = 100)
    parser.add_argument('-popSize', type=int, default = 100)
    parser.add_argument('-cxPb', type=float, default = 0.8 )
    parser.add_argument('-mtPb', type=float, default = 1 )
    parser.add_argument('-indMtPb', type=int, default = 0.07 )
    parser.add_argument('-crowding', type=float, default = 0.3)
    parser.add_argument('-seed', type=int, default = 1)
    parser.add_argument('-min_active_rules', type=int, default = 15)
    parser.add_argument('-fitness_type',type=str,default='enrollment+adverse', choices=['enrollment+adverse','enrollment+size','enrollment+size+adverse'])
    
    args = parser.parse_args()
        
    working_dir = "/home/jupyter/patient_trial_matching/"
    main(args, working_dir)
