# +
# perform_matching.py

import os, torch, json, time, argparse, random, traceback
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from prepare_data import get_matching_data
from prompts import patient_prompt, trial_prompt
from utilities import convert_response2json
from models import get_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from matching import perform_features_matching, perform_values_matching, perform_eligibility_evaluation
from sentence_transformers import SentenceTransformer

def get_test_pairs():
    test_pairs = []
    with open(os.path.join(dataDir, "benchmark_final.json"), 'r') as f:
        full_data = json.load(f)
    for dataset, trials in full_data.items():
        for trial_id, patients in trials.items():
            for patient_id, result in patients.items():
                test_pairs.append((trial_id, f"{patient_id}"))
                
    return test_pairs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark-style patient-trial pair matching.")
    parser.add_argument('--model_id', type=str, default=None, help="Model ID to use.")
    parser.add_argument('--top_k', type=int, required=True, help="top_k feature to select")
    parser.add_argument('--evaluation', type=str, required=True, choices=['llm', 'llm+nli', 'llm+rag', 'nli'], help="Evaluation agent to use")
    parser.add_argument('--dataset', type=str, required=True, choices=["colorectal_cancer", "lung_cancer", "melanoma", "prostate_cancer", "breast_cancer"])
    parser.add_argument('--matching_threshold', type=float, default=0.5, required=True, help="feature thresholding")
    parser.add_argument('--gpu', type=str, required=True)

    args = parser.parse_args()
    torch.cuda.empty_cache()

    random.seed(1)
    torch.manual_seed(1)
    device = args.gpu

    workingDir = "/home/jupyter/patient_trial_matching/"
    dataDir = os.path.join(workingDir, "data")
    dataset_dir = os.path.join(dataDir, args.dataset)
    outputDir = os.path.join(workingDir, "results", args.dataset)
    os.makedirs(outputDir, exist_ok=True)


    ground_truth = get_test_pairs()

    tokenizer, model, client = None, None, None
    client = get_model('gpt', None, device)
    if args.model_id and "gpt" not in args.model_id:
        tokenizer, model = get_model(args.model_id, os.path.join(workingDir, "scripts/models"), device)
        args.model_id = args.model_id.split("/")[-1]
    else:
        args.model_id = "NA"
    
    log_path = os.path.join(outputDir, f"{args.dataset}_benchmark_log_{args.matching_threshold}_{args.evaluation}_{args.model_id}.txt")
    with open(log_path, "w") as log_file:
        log_file.write(f"Benchmark matching started for {args.dataset}\n")
    
    if "rag" in args.evaluation:
        from matching import MedRAG
        medrag = MedRAG(
            llm_name="meta-llama/Llama-3.2-3B-Instruct",
            rag=True,
            retriever_name="MedCPT",
            corpus_name="MedText",
            corpus_cache=True,
            HNSW=True,
            db_dir="/data/shelton/RAG/corpus",
            device=device
        )
    else:
        medrag = None

    nli_tokenizer, nli_model = (None, None)
    if "nli" in args.evaluation:
        nli_tokenizer, nli_model = get_model("razent/SciFive-large-Pubmed_PMC-MedNLI", os.path.join(workingDir, "scripts/models"), device)

    semanticEncoder = SentenceTransformer("multi-qa-mpnet-base-cos-v1", device="cuda")

    patient_data = pd.read_csv(os.path.join(dataset_dir, "structured_patient_data.csv"))
    patient_data["patient_id"] = patient_data["patient_id"].astype(str)
    inclusion_trial_data = pd.read_csv(os.path.join(dataset_dir, "structured_trial_data_gpt-4.1-2025-04-14_inclusion_application.csv"))
    exclusion_trial_data = pd.read_csv(os.path.join(dataset_dir, "structured_trial_data_gpt-4.1-2025-04-14_exclusion_application.csv"))
    matching_record = {}
    cumulative_time = 0
    total_count = 0
    begin_time = time.time()

    for trial_id, patient_id in tqdm(ground_truth):
        if inclusion_trial_data[inclusion_trial_data["Trial_ID"] == trial_id].empty:
            continue
        cur_trial_inclusion = inclusion_trial_data[inclusion_trial_data["Trial_ID"] == trial_id].dropna(axis=1, how='all').drop("Trial_ID", axis=1)
        cur_trial_inclusion.columns = [col.split(": ")[1].lower() if ": " in col else col.lower() for col in cur_trial_inclusion.columns]
        
        cur_trial_exclusion = exclusion_trial_data[exclusion_trial_data["Trial_ID"] == trial_id].dropna(axis=1, how='all').drop("Trial_ID", axis=1)
        cur_trial_exclusion.columns = [col.split(": ")[1].lower() if ": " in col else col.lower() for col in cur_trial_exclusion.columns]
        
        if patient_data[patient_data["patient_id"] == patient_id].empty:
            continue
        cur_patient = patient_data[patient_data["patient_id"] == patient_id].dropna(axis=1, how='all').drop("patient_id", axis=1)
        cur_patient.columns = [col.split(": ")[1].lower() if ": " in col else col.lower() for col in cur_patient.columns]
        cur_patient = cur_patient.applymap(lambda x: x.lower() if isinstance(x, str) else x)

        patient_start_time = time.time()
        matching_features = perform_features_matching(
            cur_patient, cur_trial_inclusion, cur_trial_exclusion, "semantic-search",
            biEncoder=None, crossEncoder=None,
            semanticEncoder=semanticEncoder, client=client, top_k=args.top_k
        )

        matching_results = perform_values_matching(
            args.model_id, model, tokenizer,
            nli_model, nli_tokenizer,
            client, device,
            cur_patient, cur_trial_inclusion, cur_trial_exclusion,
            matching_features, rag=medrag,
            evaluation=args.evaluation,
            matching_threshold=args.matching_threshold,
            temperature = 0
        )

        matching_results = perform_eligibility_evaluation(matching_results)
        matching_record.setdefault(trial_id, {})[patient_id] = matching_results

        patient_time = time.time() - patient_start_time
        cumulative_time += patient_time
        total_count += 1
        avg_time = cumulative_time / total_count

        with open(log_path, "a") as log_file:
            log_file.write(f"{patient_id} x {trial_id} | {patient_time:.2f}s | Avg: {avg_time:.2f}s\n")

        with open(os.path.join(outputDir, f"{args.dataset}_benchmark_{args.matching_threshold}_{args.evaluation}_{args.model_id}_results.json"), 'w') as f:
            json.dump(matching_record, f, indent=4, ensure_ascii=False)


    with open(log_path, "a") as log_file:
        log_file.write(f"\nDone. Total time: {(time.time() - begin_time)/3600:.2f} hrs\n")

