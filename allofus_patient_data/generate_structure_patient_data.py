import os
import re
import gc
import pandas as pd
from datetime import datetime, timezone

def convert_to_age(time_stamp):
    try:
        time_stamp = datetime.fromisoformat(time_stamp)
        today = datetime.now(timezone.utc)
        return today.year - time_stamp.year - ((today.month, today.day) < (time_stamp.month, time_stamp.day))
    except:
        return None

# Process only one study for now
for study_name in ["breast_cancer"]:
    data_dir = os.path.join("/home/jupyter/patient_trial_matching/data/", study_name)
    patient_dict = {}

    print(f"{study_name} Processing...")

    ### Process Demographic Data ###
    print("Processing demographic data...")
    for chunk in pd.read_csv(os.path.join(data_dir, "person.csv"), chunksize=10000):
        for _, row in chunk.iterrows():
            patient_id = row['person_id']
            gender, ethnicity = row["sex_at_birth"], row["race"]

            if gender not in ["Not male, not female, prefer not to answer, or skipped", "No matching concept"]:
                if ethnicity not in ["PMI: Skip", "None Indicated", "None of these", "More than one population", "I prefer not to answer", "Another single population"]:
                    patient_dict[patient_id] = {
                        "demographic: gender": gender,
                        "demographic: ethnicity": ethnicity,
                        "numerical: age (years)": convert_to_age(row["date_of_birth"])
                    }
    gc.collect()
    print("Finished processing demographic data")

    ### Process Survey Data ###
    print("Processing survey data...")
    for chunk in pd.read_csv(os.path.join(data_dir, "survey.csv"), chunksize=10000):
        for _, row in chunk.iterrows():
            patient_id = row['person_id']
            if patient_id in patient_dict:
                key = f"{row['question']} {row['answer']}"
                patient_dict[patient_id][key] = "descriptive"
    gc.collect()
    print("Finished processing survey data")

    ### Process Drug Data ###
    print("Processing drug data...")
    for chunk in pd.read_csv(os.path.join(data_dir, "drug.csv"), chunksize=10000):
        for _, row in chunk.iterrows():
            patient_id = row['person_id']
            if patient_id in patient_dict:
                patient_dict[patient_id][row['standard_concept_name']] = "descriptive"
    gc.collect()
    print("Finished processing drug data")

    ### Process Procedure Data ###
    print("Processing procedure data...")
    for chunk in pd.read_csv(os.path.join(data_dir, "procedure.csv"), chunksize=10000):
        for _, row in chunk.iterrows():
            patient_id = row['person_id']
            if patient_id in patient_dict:
                patient_dict[patient_id][row['standard_concept_name']] = "descriptive"
    gc.collect()
    print("Finished processing procedure data")

    ### Process Condition Data ###
    print("Processing condition data...")
    for chunk in pd.read_csv(os.path.join(data_dir, "condition.csv"), chunksize=10000):
        for _, row in chunk.iterrows():
            patient_id = row['person_id']
            if patient_id in patient_dict:
                key = f"condition: {row['standard_concept_name']}"
                patient_dict[patient_id][key] = "present"
    gc.collect()
    print("Finished processing condition data")

    ### Process Measurement Data ###
    print("Processing measurement data...")
    for chunk in pd.read_csv(os.path.join(data_dir, "measurement.csv"), chunksize=10000):
        for _, row in chunk.iterrows():
            patient_id = row['person_id']
            if patient_id in patient_dict:
                measurement_name = re.sub(r'\s*\[.*?\]\s*', ' ', row['standard_concept_name']).strip()
                value_number = row['value_as_number']
                value_concept = row['value_as_concept_name']
                unit = row['unit_concept_name']

                if not pd.isna(measurement_name):
                    if pd.isna(value_number) or value_number == "No matching concept":
                        if not (pd.isna(value_concept) or value_concept == "No matching concept"):
                            try:
                                value_concept = float(value_concept)
                                patient_dict[patient_id][f"numerical: {measurement_name} ({unit})"] = value_concept
                            except ValueError:
                                key = f"{measurement_name} ({unit}) {value_concept}"
                                patient_dict[patient_id][key] = "descriptive"
                    else:
                        patient_dict[patient_id][f"numerical: {measurement_name} ({unit})"] = value_number
    gc.collect()
    print("Finished processing measurement data")

    ### Merge and Save ###
    print("Merging data...")
    patient_values = [{"patient_id": pid, **features} for pid, features in patient_dict.items()]
    df = pd.DataFrame.from_records(patient_values)

    csv_file = os.path.join(data_dir, "structured_patient_data.csv")
    df.to_csv(csv_file, index=False)
    print(f"{study_name} is DONE!")
