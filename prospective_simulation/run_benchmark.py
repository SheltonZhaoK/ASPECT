

# +
import os, torch, json, time, argparse, random, traceback
import pandas as pd
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from tqdm import tqdm

from collections import defaultdict
from prepare_data import get_matching_data
from prompts import patient_prompt, trial_prompt
from utilities import convert_response2json
from models import get_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from matching import perform_features_matching, perform_values_matching, perform_full_text_matching
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from transformers import AutoConfig
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
login(token = "xxx")

#python3 run_benchmark.py

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
    torch.cuda.empty_cache()
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    device = "auto"
    
    workingDir = "/home/jupyter/patient_trial_matching/"
    dataDir = os.path.join(workingDir, "data")
    outputDir = os.path.join(workingDir, "results")
    os.makedirs(outputDir, exist_ok=True)
    test_pairs = get_test_pairs()
    
    with open(os.path.join(dataDir, "trials_info_application_cleaned.json"), 'r') as f:
        trial_data = json.load(f)

    for dataset, trial in trial_data.items():
        if dataset in ["prostate_cancer"]:
            dataset_dir = os.path.join(outputDir, dataset)
            os.makedirs(dataset_dir, exist_ok=True)

            log_file_path = os.path.join(dataset_dir, f"benchmark_log.txt")
            with open(log_file_path, "w") as log_file:
                log_file.write(f"Begin matching {dataset}\n")

            patient_data = pd.read_csv(os.path.join(dataDir, dataset, "structured_patient_data.csv"))
            patient_id_list = pd.unique(patient_data["patient_id"]).tolist()

            matching_record = {}
            model_times = {}  # track time per model for this dataset
            
            for temperature in [0.0]:
                for model_id in ["google/gemma-7b-it",
                                "ibm-granite/granite-7b-instruct",
                                 "mistralai/Mistral-7B-Instruct-v0.3",
                                 "meta-llama/Llama-3.1-8B-Instruct",
                                 "Qwen/Qwen2.5-7B-Instruct"]:

                    print(f"Starting model {model_id}_{temperature}")
                    start_time = time.time()
                    matching_record[f"{model_id}_{temperature}"] = {}

                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id, torch_dtype=torch.float16, device_map="balanced",
                        max_memory={0: "10GiB", 1: "14GiB", 2: "14GiB", 3: "14GiB"}
                    )

                    for trial_id, ec in trial.items():
                        inclusion_ec = ec["inclusion criteria"]
                        exclusion_ec = ec["exclusion criteria"]
                        trial_description = f"inclusion criteria: {inclusion_ec}\nexclusion criteria: {exclusion_ec}"

                        for patient_id in patient_id_list:
                            print((dataset, temperature, model_id, trial_id, f"{patient_id}"))
                            print((trial_id, f"{patient_id}"))
                            if (trial_id, f"{patient_id}") in test_pairs:
                                print((dataset, temperature, model_id, trial_id, f"{patient_id}"))
                                cur_patient = patient_data[patient_data["patient_id"] == patient_id].dropna(axis=1, how='all').drop("patient_id", axis=1)
                                cur_patient.columns = [col.split(": ")[1].lower() if ": " in col else col.lower() for col in cur_patient.columns]
                                cur_patient = cur_patient.applymap(lambda x: x.lower() if isinstance(x, str) else x)
                                row = cur_patient.iloc[0]
                                patient_description = "; ".join(f"{col}: {val}" for col, val in row.items() if pd.notnull(val))

                                eligibility, output = perform_full_text_matching(
                                                                            model_id=model_id,
                                                                            model=model,
                                                                            tokenizer=tokenizer,
                                                                            patient=patient_description,
                                                                            trial=trial_description,
                                                                            temperature=temperature,
                                                                            client = None
                                                                        )

                                matching_record[f"{model_id}_{temperature}"].setdefault(trial_id, {})[patient_id] = eligibility

                                with open(log_file_path, "a") as log_file:
                                    log_file.write(f"\n{model_id} {temperature} {patient_id} {trial_id}\n{output}\n")

                                with open(os.path.join(dataset_dir, f"benchmark_results.json"), 'w') as json_file:
                                    json.dump(matching_record, json_file, indent=4, ensure_ascii=False)

                    end_time = time.time()
                    model_times[f"{model_id}_{temperature}"] = end_time - start_time
            
            # Save final matching results
            with open(os.path.join(dataset_dir, f"benchmark_results.json"), 'w') as json_file:
                json.dump(matching_record, json_file, indent=4, ensure_ascii=False)

            # Append timing information to the same log file
            with open(log_file_path, "a") as log_file:
                log_file.write("\nModel Timing Summary:\n")
                for model_id, duration in model_times.items():
                    log_file.write(f"{model_id} total time: {duration:.2f} seconds ({duration/60:.2f} minutes)\n")
            
            del model   
            del tokenizer
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
# -


