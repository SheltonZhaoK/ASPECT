import os, torch, json, time, argparse, re
import pandas as pd
from tqdm import tqdm
from prepare_data import get_matching_data, get_patient_reference
from prompts import (patient_prompt_step1, patient_prompt_step2, trial_prompt_step1, trial_prompt_step2)
from utilities import convert_response2json
from models import get_model
from llm_inference import openai_inference, llm_inference
from reference import ReferenceData
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSeq2SeqLM

if __name__ == "__main__":
    seed = 1
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    max_new_tokens = 10000

    workingDir = "/home/konghaoz/activeLearning_trialRecruit"
    dataDir = os.path.join(workingDir, "data")
    with open(os.path.join(dataDir, "trials_info_application_cleaned.json"), "r", encoding="utf-8") as file:
        data = json.loads(re.sub(r"[\x00-\x1F]+", " ", file.read()))
        with open(os.path.join(dataDir, "trials_info_application_cleaned.json"), "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
    data = json.load(open(os.path.join(dataDir, "trials_info_application_cleaned.json")))

    ##
    # Retrieve trial structured data
    ##     
    reference = ReferenceData()
    cat_reference = f"{reference.categorical_reference}"
    
    client = get_model('gpt', None)
    model_id = "gpt-4.1-2025-04-14"
    log_file_path = os.path.join(workingDir, "results", f"retrival_log/trial_{model_id}_application.log")
    with open(log_file_path, "w") as log_file:
        
        trial_times = []
        for application, trials in data.items():
            if application == "HNSCC":
                structured_trial_data_inclusion = pd.DataFrame()
                structured_trial_data_exclusion = pd.DataFrame()
                trial_id_list = []
                dataSaveDir = os.path.join(dataDir, "application", application)
                print(application)
                for trial_id, trial_info in tqdm(trials.items()):
                    print(f" {trial_id}\n")
                    start_time = time.time()
                    inclusive_criteria = f"Criterion: {trial_info['inclusion criteria'].strip()}"
                    exclusive_criteria = f"Criterion: {trial_info['exclusion criteria'].strip()}"
                    
                    inclusion_user_message = trial_prompt_step1(inclusive_criteria)
                    exclusion_user_message = trial_prompt_step1(exclusive_criteria)
                    for i, user_message in enumerate([inclusion_user_message, exclusion_user_message]):
                        ## Query the model and generate results ###
                        if "gpt" in model_id:
                            outputs = openai_inference(client, model_id, temperature = 0.0, task = "You are a helpful assistant. Your task is to accurately extract structured trial description from user inputs. Return your output in a list of dictionaries format.", prompt = user_message, tokens = max_new_tokens)
                        else:   
                            outputs = llm_inference(model, tokenizer, temperature = 0.0, task="You are a helpful assistant. Your task is to accurately extract structured trial description from user inputs. Return your output in a list of dictionaries format.", prompt = user_message, tokens = max_new_tokens)
                        log_file.write(f"Trial ID: {trial_id}, Output: {outputs}\n")

                        trial_features = convert_response2json(outputs, step_1=True)
                        trial_features = [entry["description"] for entry in trial_features]
                        
                        trial_dict = {}
                        if not len(trial_features) == 0:
                            user_message = trial_prompt_step2(trial_features, cat_reference)
                            
                            if "gpt" in model_id:
                                outputs = openai_inference(client, model_id, temperature = 0.0, task = "You are a helpful assistant. Your task is to accurately converted descriptive trial criterion to structured data. Return your output in a list of dictionaries format.", prompt = user_message, tokens = max_new_tokens)
                            else:
                                outputs = llm_inference(model, tokenizer, temperature = 0.0, task = "You are a helpful assistant. Your task is to accurately converted descriptive trial criterion to structured data. Return your output in a list of dictionaries format.", prompt = user_message, tokens = max_new_tokens)
                            log_file.write(f"Trial ID: {trial_id}, Output: {outputs}\n")

                            trial_data = convert_response2json(outputs, step_2=True)
                            
                            valid_values = {
                                "demographic": cat_reference,
                                "condition": ["present", "not present"],
                            }

                            for entry in trial_data:
                                entry_type = entry["type"]
                                entry_name = entry["name"]
                                entry_value = entry["value"]
                                
                                # if isinstance(entry_value, list) or entry_type == "list":
                                #     trial_dict[f"{entry_name} {entry_value}"] = "descriptive"
                        
                                if entry_type == "numerical":
                                    try:
                                        # Convert value to appropriate numerical type
                                        entry["value"] = float(entry_value) if "." in entry_value else int(entry_value)
                                        trial_dict[f"numerical: {entry_name}"] = entry["value"]
                                    except (ValueError, TypeError, KeyError):
                                        trial_dict[f"{entry_name} {entry_value}"] = "descriptive"
                                
                                elif entry_type in valid_values:
                                    if entry_value not in valid_values[entry_type] and entry_name != "ethnicity":
                                        trial_dict[f"{entry_name} {entry_value}"] = "descriptive"
                                    else:
                                        trial_dict[f"{entry_type}: {entry_name}"] = entry_value
                                else:
                                    trial_dict[f"{entry_name}"] = "descriptive"
                        
                        log_file.write(f"{trial_dict}\n")
                        if i == 0:
                            structured_trial_data_inclusion = pd.concat([structured_trial_data_inclusion, pd.DataFrame([trial_dict])], ignore_index=True)
                            structured_trial_data_inclusion.to_csv(f"{dataSaveDir}/structured_trial_data_{model_id}_inclusion_application.csv")
                        else:
                            structured_trial_data_exclusion = pd.concat([structured_trial_data_exclusion, pd.DataFrame([trial_dict])], ignore_index=True)
                            structured_trial_data_exclusion.to_csv(f"{dataSaveDir}/structured_trial_data_{model_id}_exclusion_application.csv")

                    trial_id_list.append(trial_id)
                    generation_time = time.time() - start_time
                    trial_times.append(generation_time)
                    log_file.write(f"Trial ID: {trial_id}, Generation Time: {generation_time:.2f} seconds\n\n")

                structured_trial_data_inclusion['Trial_ID'] = trial_id_list
                structured_trial_data_inclusion.set_index('Trial_ID', inplace=True)
                structured_trial_data_inclusion.to_csv(f"{dataSaveDir}/structured_trial_data_{model_id}_inclusion_application.csv")

                structured_trial_data_exclusion['Trial_ID'] = trial_id_list
                structured_trial_data_exclusion.set_index('Trial_ID', inplace=True)
                structured_trial_data_exclusion.to_csv(f"{dataSaveDir}/structured_trial_data_{model_id}_exclusion_application.csv")

        log_file.write(f"\nAverage Trial Generation Time: {sum(trial_times) / len(trial_times):.2f} seconds\n")        
        