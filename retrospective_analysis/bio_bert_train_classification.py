import os, re
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import pandas as pd
from transformers import AutoTokenizer, EarlyStoppingCallback
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def downsample_to_minority(X: pd.DataFrame,
                           y: pd.DataFrame,
                           label_col: str = "Y/N",
                           random_state: int = 1):
    """
    Make majority class the same size as the minority class (random under-sampling).
    Returns balanced X, y.
    """
    # Split indices by label
    pos_idx = y[y[label_col] == 1].index
    neg_idx = y[y[label_col] == 0].index

    n_min = min(len(pos_idx), len(neg_idx))          # minority size

    # --- sample *from the DataFrame*, then pull .index ---
    if len(pos_idx) > n_min:
        pos_idx = y.loc[pos_idx].sample(
            n=n_min, random_state=random_state).index
    if len(neg_idx) > n_min:
        neg_idx = y.loc[neg_idx].sample(
            n=n_min, random_state=random_state).index

    balanced_idx = pos_idx.union(neg_idx)

    # Shuffle once so batches are mixed
    balanced_idx = balanced_idx.to_series().sample(
        frac=1, random_state=random_state).index

    return X.loc[balanced_idx], y.loc[balanced_idx]

class TrialRecruitDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
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


def get_training_data(application, phase, dataset = "Cancer OR Malignant Neoplasm OR Neoplasm OR Malignant OR Malignant Tumour"):
    if not phase == "all":
        dataDir = os.path.join("/home/konghaoz/activeLearning_trialRecruit/data", application)
        train_x = pd.read_csv(os.path.join(dataDir, f"Phase{phase}", "train_x.csv"))
        train_y = pd.read_csv(os.path.join(dataDir, f"Phase{phase}", "train_y.csv"))
        # Filter by cancer trial IDs
        cancerData = pd.read_csv(f"/home/konghaoz/trialDesigner/results/retrieved_trials_{dataset}.csv")
        cancer_trialID = cancerData["NCT ID"].to_list()
        
        train_x = train_x[train_x['Unnamed: 0'].isin(cancer_trialID)][["eligibility/criteria/textblock"]]
        train_y = train_y.loc[train_x.index][["Y/N"]]

        # ✅ Split into inclusion & exclusion columns
        split_cols = train_x['eligibility/criteria/textblock'].apply(split_inclusion_exclusion)
        train_x = pd.concat([train_x, split_cols], axis=1)
    return train_x, train_y

for application in ["mortality", "adverse_event"]:
    output_dir = f"/home/konghaoz/activeLearning_trialRecruit/results/{application}"
    for phase in [3]:
        train_x, train_y = get_training_data(application, phase)
        train_x, eval_x, train_y, eval_y = train_test_split(
            train_x,
            train_y,
            test_size=0.2, 
            random_state=1,
        )

        # train_x, train_y = downsample_to_minority(train_x, train_y) #### balancinggggggg
        tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
        model = AutoModelForSequenceClassification.from_pretrained('dmis-lab/biobert-v1.1', num_labels=2)

        train_encoded = tokenizer(
            train_x['inclusion_text'].tolist(),
            text_pair = train_x['exclusion_text'].tolist(),
            truncation=True,
            padding='longest',
            max_length=512,
            return_tensors='pt'
        )

        eval_encoded = tokenizer(
            eval_x['inclusion_text'].tolist(),
            text_pair = eval_x['exclusion_text'].tolist(),
            truncation=True,
            padding='longest',
            max_length=512,
            return_tensors='pt'
        )

        train_dataset = TrialRecruitDataset(train_encoded, train_y["Y/N"].tolist())
        eval_dataset = TrialRecruitDataset(eval_encoded, eval_y["Y/N"].tolist())

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,          # OK to overwrite on re-runs
            num_train_epochs=10,
            per_device_train_batch_size=32,     # 256 is huge for BioBERT; 32-64 is safer
            per_device_eval_batch_size=32,
            
            eval_strategy="epoch",        # run eval each epoch
            logging_strategy="epoch",           # log every N steps
            logging_steps=1,                   # << choose a cadence that shows progress
            save_strategy="epoch",              # save after every epoch
            save_total_limit=1,                 # keep newest 3 checkpoints
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            warmup_steps=500,
            weight_decay=0.01,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
        )
        trainer.train()
        trainer.save_model(os.path.join(output_dir, f"phase{phase}"))