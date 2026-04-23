# +
import json, os, random
import pandas as pd

def get_patient_reference(path):
    data = json.load(open("/home/konghaoz/activeLearning_trialRecruit/data/patientFeatureReference.json"))
    formatted_string = ", ".join(f"{key}({value})" for category in data.values() for key, value in category.items()).lower()
    return formatted_string

def get_matching_data(dataList, saveDir, seed=None):
    if seed is not None:
        random.seed(seed)
        
    # Check if files already exist
    if os.path.isfile(os.path.join(saveDir, "labels.csv")):
        labels = pd.read_csv(os.path.join(saveDir, "labels.csv"))
        patients_info = pd.read_csv(os.path.join(saveDir, "patients_info.csv"))
        with open(os.path.join(saveDir, "trials_info.json"), 'r') as f:
            trials_info = json.load(f)
    else:
        labels = pd.DataFrame(columns=['Patient_ID', 'Trial_ID', "Eligibility"])
        patients_info = pd.DataFrame(columns=['Patient_ID', 'Description'])
        trials_info = {}

        for dataset in dataList:
            data = json.load(open(os.path.join(saveDir, f"{dataset}/retrieved_trials.json")))
            for patient in data:
                patient_id = patient["patient_id"]
                patient_description = patient["patient"]
                
                patients_info.loc[len(patients_info)] = [patient_id, patient_description]

                # Process each eligibility (0, 1, 2)
                for eligibility in range(3):
                    if dataset == "sigir" and eligibility == 1:
                        continue
                    trials_list = patient[str(eligibility)]
                    
                    # Sample at most 50 trials for each eligibility
                    if len(trials_list) > 2:
                        trials_list = random.sample(trials_list, 2)  # Randomly sample 50 trials
                    else:
                        trials_list = trials_list  # Use all trials if less than or equal to 50
                    
                    # Process each trial
                    for trial in trials_list:
                        trial_id = trial["NCTID"]
                        trials_info[trial_id] = trial
                        trials_info[trial_id].pop("NCTID", None)  # Remove 'NCTID' from the trial data

                        labels.loc[len(labels)] = [patient_id, trial_id, eligibility]

        # Save processed data to CSV and JSON files
        labels.to_csv(os.path.join(saveDir, "labels.csv"), index=False)
        patients_info.to_csv(os.path.join(saveDir, "patients_info.csv"), index=False)

        with open(os.path.join(saveDir, "trials_info.json"), 'w') as f:
            json.dump(trials_info, f, indent=4)
    
    return labels, patients_info, trials_info
