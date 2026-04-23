import os, re, json
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import torch

import pandas as pd
import torch.nn as nn
import numpy as np 

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, EarlyStoppingCallback
from torch.utils.data import TensorDataset, Dataset
from sklearn.model_selection import train_test_split

class HuberTrainer(Trainer):
    def __init__(self, *args, delta=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.huber_loss = nn.HuberLoss(delta=delta)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.view(-1)
        loss = self.huber_loss(logits, labels.float())
        return (loss, outputs) if return_outputs else loss

class TrialRecruitDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.as_tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

def split_inclusion_exclusion(text):
    """
    Split the eligibility textblock into inclusion and exclusion strings.
    Handles variants like 'Key inclusion criteria', extra spaces, and line breaks.
    """
    # Use regex, tolerant to variants and line breaks
    match = re.search(
        r"(Inclusion Criteria|Key inclusion criteria|Inclusion)\s*[:\-]?\s*(.*?)\s*(Exclusion Criteria|Key exclusion criteria|Exclusion)\s*[:\-]?\s*(.*)",
        text,
        re.DOTALL | re.IGNORECASE
    )
    if match:
        inclusion = match.group(2).strip()
        exclusion = match.group(4).strip()
    else:
        # If there's no match, assume all is inclusion, exclusion empty
        inclusion = text.strip()
        exclusion = ""
    return pd.Series({'inclusion_text': f"inclusion criteria: {inclusion}", 'exclusion_text': f"exclusion criteria: {exclusion}"})


def get_training_data(application, dataset="Cancer OR Malignant Neoplasm OR Neoplasm OR Malignant OR Malignant Tumour", max_percentile=100):
    data = pd.read_csv(f"/home/konghaoz/trialDesigner/results/retrieved_trials_{dataset}.csv")
    data["Start Date"] = pd.to_datetime(data["Start Date"], errors="coerce", utc=True)
    data["Primary Completion Date"] = pd.to_datetime(data["Primary Completion Date"], errors="coerce", utc=True)
    data["Enrollment Count"] = pd.to_numeric(data["Enrollment Count"], errors="coerce")
    data["Site Count"] = data["Locations"].fillna("").apply(
        lambda s: 0 if s.strip() == "" else s.count(";") + 1
    )
    data["Months"] = (data["Primary Completion Date"] - data["Start Date"]).dt.days / 30.44
    data["EPSM"] = data["Enrollment Count"] / (
        data["Site Count"].replace({0: np.nan}) * data["Months"].replace({0: np.nan})
    )
    
    data = data.dropna(subset=["Eligibility Criteria", application])
    cutoff = np.nanpercentile(data["Enrollment Count"].astype(float), max_percentile)
    data = data[data["Enrollment Count"] <= cutoff]

    train_x = data[["Eligibility Criteria"]]
    split_cols = train_x["Eligibility Criteria"].apply(split_inclusion_exclusion)
    train_x = pd.concat([train_x, split_cols], axis=1)
    train_y = data[[application]]
    return train_x, train_y

for application in ["Enrollment Count", "Site Count", "Months", "EPSM"]:
    output_dir = f"/home/konghaoz/activeLearning_trialRecruit/results/{application}"
    os.makedirs(output_dir, exist_ok=True)
    train_x, train_y = get_training_data(application)
    
    train_x, test_x, train_y, test_y = train_test_split(
        train_x,
        train_y,
        test_size=0.2, 
        random_state=1,
    )

    y_raw   = train_y[application].astype(float).values
    mu, sd  = y_raw.mean(), y_raw.std()
    y_z     = ((y_raw - mu) / sd).tolist()              
    with open(os.path.join(output_dir, "label_stats.json"), "w") as f:
        json.dump({"mu": mu, "sd": sd}, f)

    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
    model = AutoModelForSequenceClassification.from_pretrained('dmis-lab/biobert-v1.1', num_labels=1)
    model.config.problem_type = "regression"

    train_encoded = tokenizer(
        train_x['inclusion_text'].tolist(),
        text_pair = train_x['exclusion_text'].tolist(),
        truncation=True,
        padding='longest',
        max_length=512,
        return_tensors='pt'
    )

    train_dataset = TrialRecruitDataset(train_encoded, y_z)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,          # OK to overwrite on re-runs
        num_train_epochs=10,
        per_device_train_batch_size=128,     # 256 is huge for BioBERT; 32-64 is safer
        per_device_eval_batch_size=128,
        
        # eval_strategy="epoch",        # run eval each epoch
        logging_strategy="epoch",           # log every N steps
        logging_steps=1,                   # << choose a cadence that shows progress
        save_strategy="epoch",              # save after every epoch
        save_total_limit=1,                 # keep newest 3 checkpoints
        # load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=500,
        weight_decay=0.01,
    )
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    # )

    trainer = HuberTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    delta=1.0 
    )

    trainer.train()
    trainer.save_model(os.path.join(output_dir, "model_10e"))