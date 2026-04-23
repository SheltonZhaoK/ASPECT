# +
import os, torch, json, time, argparse, random, traceback
import pandas as pd
from tqdm import tqdm

from collections import defaultdict
from sklearn.model_selection import train_test_split
from prepare_data import get_matching_data
from prompts import patient_prompt, trial_prompt
from utilities import convert_response2json
from models import get_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from matching import perform_features_matching, perform_values_matching, perform_eligibility_evaluation
from sentence_transformers import SentenceTransformer

#python3 perform_matching.py --top_k 1 --evaluation nli --dataset prostate_cancer --gpu cuda:0 --matching_threshold 0.5 --partition 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to extract structured data using different models.")
    parser.add_argument('--model_id', type=str, default = None, help="Model ID to use (e.g., 'gpt-4o-2024-08-06').")
    parser.add_argument('--top_k', type=int, required=True, help="top_k feature to select")
    parser.add_argument('--evaluation', type=str, required=True, choices=['llm', 'llm+nli', 'llm+rag', 'nli'], help="Evaluation agent to use")
    parser.add_argument('--dataset', type=str, required=True, choices=["colorectal_cancer", "lung_cancer", "melanoma", "prostate_cancer", "breast_cancer"])
    parser.add_argument('--partition', type=int, required=True)
    parser.add_argument('--matching_threshold', type=float, default = 0.5, required=True, help="feature thresholding")
    parser.add_argument('--gpu', type=str, required=True)
    
    args = parser.parse_args()
    torch.cuda.empty_cache()
    
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    device =args.gpu
    
    top_k = args.top_k
    workingDir = "/home/jupyter/patient_trial_matching/"
    models_dir = os.path.join(workingDir,"scripts/models")
    
    tokenizer, model, client = None, None, None
    client = get_model('gpt', None, device)
    if not args.model_id is None:
        if not "gpt" in args.model_id:
            tokenizer, model = get_model(args.model_id, models_dir, device)
            args.model_id = args.model_id.split("/")[1]
    else:
        args.model_id = "NA"
    
    medrag = None
    if "rag" in args.evaluation:
        medrag = MedRAG(llm_name="meta-llama/Llama-3.2-3B-Instruct", rag=True, retriever_name="MedCPT", corpus_name="MedText", corpus_cache=True, HNSW=True, db_dir = "/data/shelton/RAG/corpus", device = device)
    
    nli_tokenizer, nli_model = None, None
    if "nli" in args.evaluation:
        nli_tokenizer, nli_model = get_model("razent/SciFive-large-Pubmed_PMC-MedNLI", models_dir, device)
    semanticEncoder = SentenceTransformer("multi-qa-mpnet-base-cos-v1", device=device)

    dataDir = os.path.join(workingDir, "data", args.dataset)
    outputDir = os.path.join(workingDir, "results", args.dataset)
    os.makedirs(outputDir, exist_ok=True)
    log_file_path = os.path.join(outputDir, f"{args.dataset}_preliminary_time_{args.matching_threshold}_{args.evaluation}_{args.model_id}_partition{args.partition}.txt")
    with open(log_file_path, "w") as log_file:
        log_file.write(f"Begin matching {args.dataset}\n")
        
    patient_data = pd.read_csv(os.path.join(dataDir, "structured_patient_data.csv"))
    if args.dataset == "breast_cancer":
        patient_data = patient_data[patient_data["demographic: gender"].str.lower() == "female"]
    elif args.dataset == "prostate_cancer":
        patient_data = patient_data[patient_data["demographic: gender"].str.lower() == "male"]
    else:
        patient_data = patient_data
        
    # === Partition patient data ===
    num_partitions = 12
    patient_ids = patient_data["patient_id"].tolist()

    # Split into roughly equal chunks
    chunk_size = len(patient_ids) // num_partitions
    remainder = len(patient_ids) % num_partitions

    start_idx = (args.partition - 1) * chunk_size + min(args.partition - 1, remainder)
    end_idx = start_idx + chunk_size + (1 if args.partition <= remainder else 0)
    
    patient_data = patient_data.iloc[start_idx:end_idx].reset_index(drop=True)
    patient_id = patient_data["patient_id"].tolist()
    
    inclusion_trial_data = pd.read_csv(os.path.join(dataDir, "structured_trial_data_gpt-4.1-2025-04-14_inclusion_application.csv"))
    exclusion_trial_data = pd.read_csv(os.path.join(dataDir, "structured_trial_data_gpt-4.1-2025-04-14_exclusion_application.csv"))
    trial_id = pd.unique(inclusion_trial_data["Trial_ID"]).tolist()

    cumulative_time = 0
    total_count = 0
    begin_time = time.time()
    matching_record = {}
    match_mode = "semantic-search"
    for trial in tqdm(trial_id):
        cur_trial_inclusion = inclusion_trial_data[inclusion_trial_data["Trial_ID"] == trial].dropna(axis=1, how='all').drop("Trial_ID", axis=1)
        cur_trial_inclusion.columns = [col.split(": ")[1].lower() if ": " in col else col.lower() for col in cur_trial_inclusion.columns]
        
        cur_trial_exclusion = exclusion_trial_data[exclusion_trial_data["Trial_ID"] == trial].dropna(axis=1, how='all').drop("Trial_ID", axis=1)
        cur_trial_exclusion.columns = [col.split(": ")[1].lower() if ": " in col else col.lower() for col in cur_trial_exclusion.columns]
        matching_record[trial] = {}

        for patient in tqdm(patient_id):
            print("\n", args.dataset, args.partition, trial, patient, "\n")
            patient_start_time = time.time()
            try:
                cur_patient = patient_data[patient_data["patient_id"] == patient].dropna(axis=1, how='all').drop("patient_id", axis=1)
                cur_patient.columns = [col.split(": ")[1].lower() if ": " in col else col.lower() for col in cur_patient.columns]
                cur_patient = cur_patient.applymap(lambda x: x.lower() if isinstance(x, str) else x)
                
                matching_features = perform_features_matching(cur_patient, cur_trial_inclusion, cur_trial_exclusion, match_mode, biEncoder = None, crossEncoder = None, semanticEncoder = semanticEncoder, client = client, top_k = args.top_k) #bi-encoder, cross-encoder, semantic-search
                matching_results = perform_values_matching(args.model_id, model, tokenizer, nli_model, nli_tokenizer, client, device, cur_patient, cur_trial_inclusion, cur_trial_exclusion, matching_features, rag=medrag, evaluation=args.evaluation, matching_threshold=args.matching_threshold, temperature=0.0)
                matching_results = perform_eligibility_evaluation(matching_results)
                matching_record[trial][patient] = matching_results

                patient_time = time.time() - patient_start_time
                cumulative_time += patient_time
                total_count += 1
                cumulative_avg_time = cumulative_time / total_count

                with open(log_file_path, "a") as log_file:
                    log_file.write(f"{patient} and {trial} DONE | Total Count {total_count}\n")
                    log_file.write(f"Individual time: {patient_time:.2f} sec\n")
                    log_file.write(f"Cumulative time: {cumulative_time:.2f} sec ({cumulative_time/3600:.2f} hrs)\n")
                    log_file.write(f"Cumulative average time: {cumulative_avg_time:.2f} sec\n\n")

            except Exception as e:
                tb_str = traceback.format_exc()
                error_message = f"Error processing patient {patient} in trial {trial}:\n{tb_str}\n"
                print(error_message)
                with open(log_file_path, "a") as log_file:
                    log_file.write(error_message)
                continue

        with open(os.path.join(outputDir, f"largeScale_{args.dataset}_{args.matching_threshold}_{args.evaluation}_{args.model_id}_results_partition{args.partition}.json"), 'w') as json_file:
            json.dump(matching_record, json_file, indent=4, ensure_ascii=False)

    with open(log_file_path, "a") as log_file:
        log_file.write(f"Evaluation time: {(time.time()-begin_time)/3600:.2f} hrs")
