# +
import os, random, matplotlib
random.seed(1)
import json
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, balanced_accuracy_score, cohen_kappa_score
)
matplotlib.rcParams['svg.fonttype'] = 'none'

# ========== STEP 1: Helpers ==========

def read_results(data):
    patient_ids, trial_ids, eligibilities = [], [], []
    for trial_id, patient in data.items():
        for patient_id, patient_info in patient.items():
            if isinstance(patient_info, dict):
                eligibility = patient_info.get('eligibility', None)
            else:
                eligibility = patient_info
            if eligibility in ["N/A", None]:
                eligibility = 0
            patient_ids.append(patient_id)
            trial_ids.append(trial_id)
            eligibilities.append(int(eligibility) if eligibility != -1 else -1)
    return pd.DataFrame({'patient_id': patient_ids, 'trial_id': trial_ids, 'eligibility': eligibilities})

def update_or_insert(df, row):
    mask = (df["Method"] == row["Method"]) & (df["Dataset"] == row["Dataset"])
    if df[mask].empty:
        new_row_df = pd.DataFrame([row])
        df = pd.concat([df, new_row_df], ignore_index=True)
    else:
        for key in row:
            df.loc[mask, key] = row[key]
    return df


# ========== STEP 2: Load Ground Truth ==========

ground_truth = {}
# for file in ["test_final.json", "test_final_2.json"]:
for file in ["test_label_final.json"]:
    with open(f"/home/jupyter/patient_trial_matching/data/{file}", 'r') as f:
        full_data = json.load(f)
    for dataset, trials in full_data.items():
        for trial_id, patients in trials.items():
            for patient_id, result in patients.items():
                ground_truth[(trial_id, f"{patient_id}")] = result["eligibility"]
                

                
# ========== STEP 3: Process Evaluation Results ==========


# thresholds = [0.5]
thresholds = [0.5]
for threshold in thresholds:
    metrics_df = []
    
    root_dir = "/home/jupyter/patient_trial_matching/results"
    # Proposed architectures
    for dataset in full_data.keys():
        for method in ["nli"]:
            model = "NA" if method == "nli" else "Qwen2.5-7B-Instruct"
            path = os.path.join(root_dir, dataset, f"{dataset}_benchmark_{threshold}_{method}_{model}_results.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                result = json.load(f)
            df = read_results(result)
#                 df = df[df['eligibility'] != -1]

            y_true, y_pred = [], []
            for _, row in df.iterrows():
                key = (row["trial_id"], f"{row['patient_id']}")
                if key in ground_truth and row["eligibility"] in [0, 1]:
                    y_true.append(ground_truth[key])
                    y_pred.append(row["eligibility"])

            if not y_true:
                continue

            all_prediction = df['eligibility'].tolist()
            inclusion_count = sum(all_prediction)
            inclusion_rate = round(inclusion_count / len(all_prediction), 3) if all_prediction else 0

            accuracy = round(accuracy_score(y_true, y_pred), 3)
#             precision = round(precision_score(y_true, y_pred), 3)
#             negative_precision = round(precision_score(y_true, y_pred, pos_label=0), 3)
#             recall = round(recall_score(y_true, y_pred), 3)
            f1 = round(f1_score(y_true, y_pred, average='macro'), 3)
            weighted_f1 = f1_score(y_true, y_pred, average='weighted')
#             balanced_acc = balanced_accuracy_score(y_true, y_pred)
            kappa = cohen_kappa_score(y_true, y_pred)

#             cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
# #             specificity = round(cm[0, 0] / (cm[0, 0] + cm[0, 1]), 3) if cm.shape == (2, 2) and (cm[0, 0] + cm[0, 1]) > 0 else None

            metrics_df.append({
                "Method": f"Proposed Architecture {method}",
                "Dataset": dataset,
                "Accuracy": accuracy,
#                 "Precision": precision,
#                 "Negative Precision": negative_precision,
#                 "Sensitivity": recall,
                "F1": f1,
#                 "Specificity": specificity,
#                 "Inclusion Count": inclusion_count,
#                 "Inclusion Rate": inclusion_rate,
                "Weighted F1": weighted_f1,
                "Cohen's Kappa": kappa,
#                 "Balanced Accuracy": balanced_acc
            })

            y_true_cm, y_pred_cm = [], []
            for _, row in df.iterrows():
                key = (row["trial_id"], f"{row['patient_id']}")
                if key in ground_truth and row["eligibility"] in [0, 1]:
                    y_true_cm.append(ground_truth[key])
                    y_pred_cm.append(row["eligibility"])

    # Benchmark models
    for dataset in full_data.keys():
        path = os.path.join(root_dir, dataset, "benchmark_results.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            benchmark_results = json.load(f)
        for model, result in benchmark_results.items():
            if "ibm-granite" in model:
                continue
            df = read_results(result)
    #         df = df[df['eligibility'] != -1]

            y_true, y_pred = [], []
            for _, row in df.iterrows():
                key = (row["trial_id"], f"{row['patient_id']}")
                if key in ground_truth and row["eligibility"] in [0, 1]:
                    y_true.append(ground_truth[key])
                    y_pred.append(row["eligibility"])

            if not y_true:
                continue

            all_prediction = df['eligibility'].tolist()
            inclusion_count = sum(all_prediction)
            inclusion_rate = round(inclusion_count / len(all_prediction), 3)

    #         accuracy = round(accuracy_score(y_true, y_pred), 3)
            accuracy = round((np.array(y_pred) == np.array(y_true)).mean(), 3)
    #         precision = round(precision_score(y_true, y_pred), 3)
    #         negative_precision = round(precision_score(y_true, y_pred, pos_label=0), 3)
    #         recall = round(recall_score(y_true, y_pred), 3)
            f1 = round(f1_score(y_true, y_pred, average='macro'), 3)
            weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    #         balanced_acc = balanced_accuracy_score(y_true, y_pred)
            kappa = cohen_kappa_score(y_true, y_pred)

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            specificity = round(cm[0, 0] / (cm[0, 0] + cm[0, 1]), 3) if cm.shape == (2, 2) and (cm[0, 0] + cm[0, 1]) > 0 else None

            metrics_df.append({
                "Method": model,
                "Dataset": dataset,
                "Accuracy": accuracy,
    #             "Precision": precision,
    #             "Negative Precision": negative_precision,
    #             "Sensitivity": recall,
                "F1": f1,
    #             "Specificity": specificity,
    #             "Inclusion Count": inclusion_count,
    #             "Inclusion Rate": inclusion_rate,
                "Cohen's Kappa": kappa,
                "Weighted F1": weighted_f1,
#                 "Balanced Accuracy": balanced_acc
            })

    # Panacea
    for dataset in full_data.keys():
        path = os.path.join(root_dir, dataset, "benchmark_result_panacea.json")
        with open(path, 'r') as file:
            data = json.load(file)

        y_pred, y_true = [], []
        patients_id, trial_ids = [], []

        for patient_id, trials in data.items():
            for trial_id, value in trials.items():
                output = value.get('output', '')

                # Extract prediction from model output
                if 'eligibility: 2)' in output:
                    pred = 1
                elif 'eligibility: 0)' in output or 'eligibility: 1)' in output:
                    pred = 0
                else:
                    pred = -1  # fallback

                key = (trial_id, str(patient_id))
                if key not in ground_truth:
                    continue  # skip unmatched entries

                label = ground_truth[key]
                if label not in [0, 1]:
                    continue

                y_pred.append(pred)
                y_true.append(label)
                patients_id.append(patient_id)
                trial_ids.append(trial_id)

        if not y_true:
            continue

        inclusion_count = sum(y_pred)
        inclusion_rate = round(inclusion_count / len(y_pred), 3)

        accuracy = round(accuracy_score(y_true, y_pred), 3)
    #     precision = round(precision_score(y_true, y_pred), 3)
    #     negative_precision = round(precision_score(y_true, y_pred, pos_label=0), 3)
    #     recall = round(recall_score(y_true, y_pred), 3)
        f1 = round(f1_score(y_true, y_pred, average='macro'), 3)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        specificity = round(cm[0, 0] / (cm[0, 0] + cm[0, 1]), 3) if (cm.shape == (2, 2) and (cm[0, 0] + cm[0, 1]) > 0) else None

        metrics_df.append({
            "Method": "Panacea-7B",
            "Dataset": dataset,
            "Accuracy": accuracy,
    #         "Precision": precision,
    #         "Negative Precision": negative_precision,
    #         "Sensitivity": recall,
            "F1": f1,
    #         "Specificity": specificity,
    #         "Inclusion Count": inclusion_count,
    #         "Inclusion Rate": inclusion_rate,
            "Cohen's Kappa": kappa,
            "Weighted F1": weighted_f1,
            "Balanced Accuracy": balanced_acc
        })

    # ========== STEP 4: Evaluation Time + Size ==========
    time_entries = [
        # breast_cancer
        {"Method": "Proposed Architecture nli", "Dataset": "breast_cancer", "Evaluation Time (minutes)": 0.04 * 60, "Model Size (Billion)": 0.738},
#         {"Method": "Proposed Architecture llm+nli", "Dataset": "breast_cancer", "Evaluation Time (minutes)": 0.17 * 60, "Model Size (Billion)": 0.738 + 7.61},
#         {"Method": "Proposed Architecture llm", "Dataset": "breast_cancer", "Evaluation Time (minutes)": 1.47 * 60, "Model Size (Billion)": 7.61},
        {"Method": "Qwen/Qwen2.5-7B-Instruct_0.0", "Dataset": "breast_cancer", "Evaluation Time (minutes)": 88.02, "Model Size (Billion)": 7.61},
        {"Method": "meta-llama/Llama-3.1-8B-Instruct_0.0", "Dataset": "breast_cancer", "Evaluation Time (minutes)": 460.33, "Model Size (Billion)": 8},
        {"Method": "google/gemma-7b-it_0.0", "Dataset": "breast_cancer", "Evaluation Time (minutes)": 20.98, "Model Size (Billion)": 7.75},
        {"Method": "mistralai/Mistral-7B-Instruct-v0.3_0.0", "Dataset": "breast_cancer", "Evaluation Time (minutes)": 685.16, "Model Size (Billion)": 7.3},
        {"Method": "Panacea-7B", "Dataset": "breast_cancer", "Evaluation Time (minutes)": 119.78, "Model Size (Billion)": 7.3},
        
        # colorectal_cancer
        {"Method": "Proposed Architecture nli", "Dataset": "colorectal_cancer", "Evaluation Time (minutes)": 0.01 * 60, "Model Size (Billion)": 0.738},
#         {"Method": "Proposed Architecture llm+nli", "Dataset": "colorectal_cancer", "Evaluation Time (minutes)": 0.06 * 60, "Model Size (Billion)": 0.738 + 7.61},
#         {"Method": "Proposed Architecture llm", "Dataset": "colorectal_cancer", "Evaluation Time (minutes)": 0.35 * 60, "Model Size (Billion)": 7.61},
        {"Method": "Qwen/Qwen2.5-7B-Instruct_0.0", "Dataset": "colorectal_cancer", "Evaluation Time (minutes)": 40.05, "Model Size (Billion)": 7.61},
        {"Method": "meta-llama/Llama-3.1-8B-Instruct_0.0", "Dataset": "colorectal_cancer", "Evaluation Time (minutes)": 200.71, "Model Size (Billion)": 8},
        {"Method": "google/gemma-7b-it_0.0", "Dataset": "colorectal_cancer", "Evaluation Time (minutes)": 27.79, "Model Size (Billion)": 7.75},
        {"Method": "mistralai/Mistral-7B-Instruct-v0.3_0.0", "Dataset": "colorectal_cancer", "Evaluation Time (minutes)": 137.53, "Model Size (Billion)": 7.3},
        {"Method": "Panacea-7B", "Dataset": "colorectal_cancer", "Evaluation Time (minutes)": 30.76, "Model Size (Billion)": 7.3},

        # lung_cancer
        {"Method": "Proposed Architecture nli", "Dataset": "lung_cancer", "Evaluation Time (minutes)": 0.06 * 60, "Model Size (Billion)": 0.738},
#         {"Method": "Proposed Architecture llm+nli", "Dataset": "lung_cancer", "Evaluation Time (minutes)": 0.3 * 60, "Model Size (Billion)": 0.738 + 7.61},
#         {"Method": "Proposed Architecture llm", "Dataset": "lung_cancer", "Evaluation Time (minutes)": 2.26 * 60, "Model Size (Billion)": 7.61},
        {"Method": "Qwen/Qwen2.5-7B-Instruct_0.0", "Dataset": "lung_cancer", "Evaluation Time (minutes)": 148.08, "Model Size (Billion)": 7.61},
        {"Method": "meta-llama/Llama-3.1-8B-Instruct_0.0", "Dataset": "lung_cancer", "Evaluation Time (minutes)": 848.50, "Model Size (Billion)": 8},
        {"Method": "google/gemma-7b-it_0.0", "Dataset": "lung_cancer", "Evaluation Time (minutes)": 20.98, "Model Size (Billion)": 7.75},
        {"Method": "mistralai/Mistral-7B-Instruct-v0.3_0.0", "Dataset": "lung_cancer", "Evaluation Time (minutes)": 1039.94, "Model Size (Billion)": 7.3},
        {"Method": "Panacea-7B", "Dataset": "lung_cancer", "Evaluation Time (minutes)": 172.94, "Model Size (Billion)": 7.3},

        # melanoma
        {"Method": "Proposed Architecture nli", "Dataset": "melanoma", "Evaluation Time (minutes)": 0.01 * 60, "Model Size (Billion)": 0.738},
#         {"Method": "Proposed Architecture llm+nli", "Dataset": "melanoma", "Evaluation Time (minutes)": 0.05 * 60, "Model Size (Billion)": 0.738 + 7.61},
#         {"Method": "Proposed Architecture llm", "Dataset": "melanoma", "Evaluation Time (minutes)": 0.47 * 60, "Model Size (Billion)": 7.61},
        {"Method": "Qwen/Qwen2.5-7B-Instruct_0.0", "Dataset": "melanoma", "Evaluation Time (minutes)": 14.33, "Model Size (Billion)": 7.61},
        {"Method": "meta-llama/Llama-3.1-8B-Instruct_0.0", "Dataset": "melanoma", "Evaluation Time (minutes)": 145.97, "Model Size (Billion)": 8},
        {"Method": "google/gemma-7b-it_0.0", "Dataset": "melanoma", "Evaluation Time (minutes)": 9.01, "Model Size (Billion)": 7.75},
        {"Method": "mistralai/Mistral-7B-Instruct-v0.3_0.0", "Dataset": "melanoma", "Evaluation Time (minutes)": 169.29, "Model Size (Billion)": 7.3},
        {"Method": "Panacea-7B", "Dataset": "melanoma", "Evaluation Time (minutes)": 34.95, "Model Size (Billion)": 7.3},

        # prostate_cancer
        {"Method": "Proposed Architecture nli", "Dataset": "prostate_cancer", "Evaluation Time (minutes)": 0.02 * 60, "Model Size (Billion)": 0.738},
#         {"Method": "Proposed Architecture llm+nli", "Dataset": "prostate_cancer", "Evaluation Time (minutes)": 0.09 * 60, "Model Size (Billion)": 0.738 + 7.61},
#         {"Method": "Proposed Architecture llm", "Dataset": "prostate_cancer", "Evaluation Time (minutes)": 0.92 * 60, "Model Size (Billion)": 7.61},
        {"Method": "Qwen/Qwen2.5-7B-Instruct_0.0", "Dataset": "prostate_cancer", "Evaluation Time (minutes)": 41.71, "Model Size (Billion)": 7.61},
        {"Method": "meta-llama/Llama-3.1-8B-Instruct_0.0", "Dataset": "prostate_cancer", "Evaluation Time (minutes)": 230.90, "Model Size (Billion)": 8},
        {"Method": "google/gemma-7b-it_0.0", "Dataset": "prostate_cancer", "Evaluation Time (minutes)": 11.97, "Model Size (Billion)": 7.75},
        {"Method": "mistralai/Mistral-7B-Instruct-v0.3_0.0", "Dataset": "prostate_cancer", "Evaluation Time (minutes)": 251.50, "Model Size (Billion)": 7.3},
        {"Method": "Panacea-7B", "Dataset": "prostate_cancer", "Evaluation Time (minutes)": 32.93, "Model Size (Billion)": 7.3}
    ]
    desired_order = ["Proposed Architecture nli","Qwen/Qwen2.5-7B-Instruct_0.0","meta-llama/Llama-3.1-8B-Instruct_0.0", "google/gemma-7b-it_0.0", "mistralai/Mistral-7B-Instruct-v0.3_0.0" , "Panacea-7B"]
    # Dedup-safe update
    metrics_df = pd.DataFrame(metrics_df)
    for row in time_entries:
        metrics_df = update_or_insert(metrics_df, row)
    print(metrics_df)

    # === Setup ===
#     color =[
#         "#28A9A1", "#C9A77C", "#F4A016",'#F6BBC6','#E71F19', "#D6594C", "#7D8995", 
#         "#E9BD27"]
#     color = [
#         "#E71F19",  # bright red
#         "#28A9A1",  # teal (cool contrast)
#         "#F4A016",  # orange
#         "#7262ac",  # purple
#         "#E9BD27",  # yellow
#         "#7D8995",  # gray (neutral break)
#         "#F6BBC6",  # soft pink
#         "#D6594C",  # muted red
#         "#C9A77C"   # beige (soft ending)
#     ]
    
    color = [
    "#28A9A1",  # teal (method 1 – fresh, modern)
    "#F4A016",  # orange (method 2 – energetic, contrasts teal)
    "#E71F19",  # red (method 3 – bold, very distinct)

    "#7262ac",  # purple (secondary, cooler)
    "#E9BD27",  # yellow (bright but less dominant after red/orange)
    "#7D8995",  # gray (neutral balance)
    "#C9A77C",  # beige (soft contrast)
    "#F6BBC6",  # pink (gentle tone)
    "#D6594C"   # muted red (wraps up with warmth)
    ]

#     # === FIGURE 1: application_test.pdf ===
#     metrics_to_plot = ["F1"]
#     print(metrics_to_plot)
#     fig, axes = plt.subplots(nrows=1, ncols=len(metrics_to_plot), figsize=(10 * len(metrics_to_plot), 5), sharex=True)
#     metric = metrics_to_plot[0]
#     bars = sns.barplot(
#         data=metrics_df,
#         x="Dataset",
#         y=metric,
#         hue="Method",
#         ax=axes,
#         palette=color, 
#         hue_order = desired_order,
#         edgecolor="black",
#         linewidth=0.5
#     )
#     axes.legend_.remove()
#     axes.set_xlabel("")
#     axes.set_ylabel(metric)
    
#     axes.spines['top'].set_visible(False)
#     axes.spines['right'].set_visible(False)
# #     axes.grid(True, axis='y')
# #     axes.set_axisbelow(True)
# #     axes.tick_params(axis='x', rotation=45)
# #         for p in bars.patches:
# #             height = p.get_height()
# #             if pd.notnull(height) and height > 0:
# #                 axes.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
# #                               ha='center', va='bottom', fontsize=8, fontweight='bold')

#     # Shared legend
#     handles, labels = axes.get_legend_handles_labels()
#     fig.legend(handles, labels, loc='lower center',
#                bbox_to_anchor=(0.5, -0.2), ncol=4)
#     plt.savefig(f"benchmark_{threshold}.svg")
#     plt.show()

#     avg_by_model = (
#         metrics_df[["Method", "Evaluation Time (minutes)", "Model Size (Billion)"]]
#         .dropna()
#         .groupby("Method", as_index=False)
#         .agg({
#             "Evaluation Time (minutes)": ["mean", "std"],
#             "Model Size (Billion)": "first"
#         })
#     )
#     avg_by_model.columns = ["Method", "Evaluation Time (minutes)", "Time STD", "Model Size (Billion)"]
#     avg_by_model["Method"] = pd.Categorical(avg_by_model["Method"], categories=desired_order, ordered=True)
#     avg_by_model = avg_by_model.sort_values("Method")

#     fig, ax1 = plt.subplots(figsize=(10, 5))
#     bars = sns.barplot(
#         data=avg_by_model,
#         x="Method",
#         y="Evaluation Time (minutes)",
#         palette=color,
#         ax=ax1,
#         edgecolor="black",
#         linewidth=0.5,
#         errorbar=None, ci=None
        
#     )

#     # Add std error bars manually
#     for bar, (_, row) in zip(bars.patches, avg_by_model.iterrows()):
#         x = bar.get_x() + bar.get_width() / 2
#         y = row["Evaluation Time (minutes)"]
#         std = row["Time STD"]
#         if pd.notna(std) and y > 0:
#             yerr_low = max(y - std, 1e-2)
#             yerr_high = y + std
#             ax1.plot([x, x], [yerr_low, yerr_high], color="black", linewidth=1.5)
#             ax1.plot([x - 0.05, x + 0.05], [yerr_high, yerr_high], color="black", linewidth=1)
#             ax1.plot([x - 0.05, x + 0.05], [yerr_low, yerr_low], color="black", linewidth=1)

#     ax1.set_yscale("log")
#     ax1.set_xlabel("")
#     ax1.set_ylim(1, 1000)
#     ax1.set_ylabel("Average Evaluation Time (mins)")
#     ax1.tick_params(axis='y')
#     ax1.tick_params(axis='x', rotation=45)
#     ax1.spines['top'].set_visible(False)
# #     ax1.spines['right'].set_visible(False)

#     # Line for model size
#     ax2 = ax1.twinx()
#     ax2.grid(False)
#     sns.lineplot(
#         data=avg_by_model,
#         x="Method",
#         y="Model Size (Billion)",
#         marker="o",
#         linewidth=1,
#         color="black",     # ← Set line and marker color to black
#         ax=ax2,
#         label="Model Size"
#     )
#     ax2.set_xlabel("")
#     ax2.set_xticks([])
#     ax2.set_ylabel("Model Size (Billion)", color="black")
#     ax2.tick_params(axis='y', labelcolor="black")
#     ax2.spines['top'].set_visible(False)
# #     ax2.spines['right'].set_visible(False)

# #     plt.tight_layout(rect=[0, 0.1, 1, 1])
#     plt.savefig("combined_model_time_size.svg", bbox_inches="tight")
#     plt.show()

baseline_method = "Proposed Architecture nli"
desired_order = [
    "Proposed Architecture nli",
    "google/gemma-7b-it_0.0",
    "Qwen/Qwen2.5-7B-Instruct_0.0",
    "meta-llama/Llama-3.1-8B-Instruct_0.0",
    "mistralai/Mistral-7B-Instruct-v0.3_0.0",
    "Panacea-7B"
]

# --- Compute average evaluation time and model size ---
avg_by_model = (
    metrics_df[["Method", "Evaluation Time (minutes)", "Model Size (Billion)"]]
    .dropna()
    .groupby("Method", as_index=False)
    .agg({"Evaluation Time (minutes)": "mean", "Model Size (Billion)": "first"})
)
avg_by_model["Method"] = pd.Categorical(avg_by_model["Method"], categories=desired_order, ordered=True)
avg_by_model = avg_by_model.sort_values("Method")

# Determine baseline and ratio
if baseline_method not in avg_by_model["Method"].values:
    print("⚠️ Baseline method not found — skipping ratio annotations.")
    baseline_time = avg_by_model["Evaluation Time (minutes)"].min()
else:
    baseline_time = avg_by_model.loc[
        avg_by_model["Method"] == baseline_method, "Evaluation Time (minutes)"
    ].values[0]
avg_by_model["x_slower"] = avg_by_model["Evaluation Time (minutes)"] / baseline_time


# --- Create figure with 3 subplots ---
fig, axes = plt.subplots(1, 3, figsize=(25, 6))
# ========== (A) F1 SCORE ==========
sns.barplot(
    data=metrics_df,
    x="Dataset",
    y="F1",
    hue="Method",
    hue_order=desired_order,
    palette=color,
    edgecolor="black",linewidth=0.5,
    ax=axes[0]
)
axes[0].set_xlabel("")
axes[0].set_ylabel("F1 Score")
axes[0].legend_.remove()  # hide legend here

# ========== (B) EVALUATION TIME ==========
bars = sns.barplot(
    data=avg_by_model,
    x="Method",
    y="Evaluation Time (minutes)",
    palette=color,
    ax=axes[1],
    edgecolor="black",linewidth=0.5
)
for bar, (_, row) in zip(bars.patches, avg_by_model.iterrows()):
    method = row["Method"]
    ratio = row["x_slower"]
    if method != baseline_method and ratio > 1.1:
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.05,
            f"×{ratio:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )
axes[1].set_yscale("log")
axes[1].set_ylim(1, 1200)
axes[1].set_title("Evaluation Time", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Minutes (log scale)")
axes[1].set_xlabel("")

# ========== (C) MODEL SIZE ==========
sns.barplot(
    data=avg_by_model,
    x="Method",
    y="Model Size (Billion)",
    palette=color,
    ax=axes[2],
    edgecolor="black",linewidth=0.5,
)
axes[2].set_ylabel("Billion Params")
axes[2].set_xlabel("")

# --- Shared Legend ---
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.08),
    ncol=3,
    frameon=False
)

for ax in axes:
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.yaxis.label.set_color('black')
    ax.xaxis.label.set_color('black')
    ax.grid(False)
    ax.tick_params(
    axis='both',
    which='major',
    direction='out',   # or 'inout' if you want both sides
    length=5,          # visible tick length
    width=1.2,         # visible tick width
    color='black',     # tick color
    labelcolor='black',
    bottom=True,       # ensure bottom ticks visible
    top=False,
    left=True,
    right=False
    )
    
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("combined_metrics.svg", bbox_inches="tight")
plt.show()

# -


