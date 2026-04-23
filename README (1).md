# ASPECT

**AmbiSPective Eligibility Criteria analysis for clinical Trials**

A data-driven framework that integrates retrospective mining of historical clinical trials with prospective simulation on real-world patient data to evaluate, quantify, and optimize eligibility criteria in Phase 3 oncology trials.

🌐 **Interactive web interface:** https://aspect-trial.com/

---

## Overview

Over half of terminated clinical trials close because of inadequate accrual, with restrictive eligibility criteria identified as a major contributing factor. ASPECT addresses this by combining two complementary views of eligibility design:

- **Retrospective trial mining** — extracts, parses, and semantically matches eligibility rules across 1,937 historical Phase 3 oncology trials from ClinicalTrials.gov. Transformer-based models predict associations between criteria and trial outcomes (enrollment, site count, duration, enrollment per site-month, mortality, adverse-event occurrence).
- **Prospective real-world simulation** — applies eligibility rules to 20,126 real-world cancer patients from the All of Us Research Program to compute criterion-level and trial-level exclusion rates stratified by age, sex, and race.
- **Multi-objective optimization** — a genetic algorithm (NSGA-III) explores the combinatorial space of eligibility rule subsets to identify Pareto-optimal configurations that balance enrollment feasibility against predicted safety.

Across 285 Phase 3 oncology trials spanning breast, colorectal, lung, melanoma, and prostate cancer, we find that contemporary eligibility criteria exclude an average of 86% of real-world patients at the trial level, and that this exclusion is concentrated in a small subset of highly restrictive rules (top 20% of rules → 71.3% of total exclusion burden; Gini = 0.691).

---

## Repository structure

```
ASPECT/
├── README.md
├── LICENSE
├── .gitignore
├── environment.yml                    ← conda environment
│
├── retrospective_analysis/            ← historical trial mining
│   ├── fetch_clinicalTrial.py         ← query ClinicalTrials.gov API v2
│   ├── retrive_relevent_ec_large_scale.py       ← SBERT + SciFive semantic matching [text]
│   ├── retrieve_relevant_ec_large_numerical.py  ← SBERT + SciFive semantic matching [numerical]
│   ├── frequency_ranking.py           ← rule frequency + clustering across trials
│   ├── frequency_ranking_numerical.py ← numerical-threshold frequency extraction
│   ├── longitudinal_analysis.py       ← temporal trends (Fig. 3C)
│   ├── enrollment_analysis.py         ← rule frequency vs. EPSM (Fig. 3B)
│   ├── bio_bert_train_classification.py ← train mortality / adverse-event classifiers
│   ├── bio_bert_train_regression.py   ← train enrollment / EPSM / duration / site regressors
│   └── bio_bert_inference.py          ← run trained models on new trials (leave-one-out)
│
├── prospective_simulation/            ← real-world patient-trial matching
│   ├── prepare_data.py                ← load & align trial + patient features
│   ├── retrieve_structure_table_application.py  ← GPT-4.1 two-step feature decomposition
│   ├── prompts.py                     ← all LLM prompts
│   ├── llm_inference.py               ← LLM wrappers (OpenAI + local)
│   ├── models.py                      ← model loaders (Llama, Qwen, Mistral, Gemma, Panacea)
│   ├── matching.py                    ← SBERT feature alignment + NLI evaluation
│   ├── perform_matching.py            ← end-to-end criterion-level matching pipeline
│   ├── run_matching_benchmark.py      ← benchmark ASPECT on clinician-annotated pairs
│   ├── run_benchmark.py               ← benchmark baseline LLMs (Llama/Qwen/Mistral/Gemma)
│   └── benchmark_evaluation.py        ← compute F1 / κ / timing vs clinician labels (Fig. 6)
│
├── multi-objective_optimization/      ← MOGA
│   ├── moga.py                        ← NSGA-III implementation (DEAP)
│   └── run_moga.py                    ← apply MOGA to a trial's eligibility set
│
├── allofus_patient_data/              ← All of Us Workbench artifacts
│   ├── Breast Cancer.ipynb            ← cohort SQL + feature extraction (run in AoU)
│   ├── Colorectal Cancer.ipynb
│   ├── Lung Cancer Final.ipynb
│   ├── Melanoma Final.ipynb
│   ├── Prostate Cancer.ipynb
│   └── generate_structure_patient_data.py
│
├── website/                           ← web-interface data preparation
│   └── prepare_website_data.py        ← aggregates rule-level stats for aspect-trial.com
│
├── data/                              ← releasable inputs
│   ├── trials_info_application_cleaned.json   ← 285 trials × structured eligibility criteria
│   ├── annotation.json                ← 289 clinician-annotated patient-trial pairs (de-identified)
│   ├── breast_cancer/
│   │   ├── structured_trial_data_gpt-4.1-2025-04-14_inclusion_application.csv
│   │   └── structured_trial_data_gpt-4.1-2025-04-14_exclusion_application.csv
│   ├── colorectal_cancer/
│   ├── lung_cancer/
│   ├── melanoma/
│   └── prostate_cancer/
│
└── results/                           ← sample pipeline outputs (subset)
    ├── all_trials_rule_exclusion_rates.csv      ← trial × rule × demographic exclusion rates
    ├── breast_cancer_rule_exclusion.json        ← per-trial, per-rule exclusion details
    ├── colorectal_cancer_rule_exclusion.json
    ├── lung_cancer_rule_exclusion.json
    ├── melanoma_rule_exclusion.json
    ├── prostate_cancer_rule_exclusion.json
    └── full_rule_summary_nli_check_0.6_numerical.json   ← laboratory threshold summary
```

> **Note on the `results/` folder:** all result files used to build the web interface (https://aspect-trial.com/) can be found in `results/`, with the exception of `full_rule_summary_nli_check_0.6.json`, which is too large for GitHub and can be regenerated by running `retrospective_analysis/retrive_relevent_ec_large_scale.py`.

---

## Installation

```bash
git clone https://github.com/<YOUR_ORG>/aspect-trial.git
cd aspect-trial

# Recommended: conda environment
conda env create -f environment.yml
conda activate aspect

# Alternative: pip
python -m venv venv
source venv/bin/activate         # on Windows: venv\Scripts\activate
pip install torch transformers sentence-transformers accelerate \
            huggingface-hub openai python-dotenv deap faiss-cpu \
            hdbscan umap-learn pandas numpy scipy scikit-learn \
            matplotlib seaborn tqdm joblib jinja2 requests
```

GPU is required for BioBERT / SBERT / SciFive / local-LLM inference. CPU-only mode works for the MOGA module and small-scale analyses.

### Environment variables

Create a `.env` file at the repository root (never commit it — `.gitignore` excludes it):

```bash
OPENAI_API_KEY=sk-...            # required for feature decomposition (GPT-4.1)
HF_TOKEN=hf_...                  # required to load gated models (e.g. Llama, Panacea)
```

---

## Data

### What's in this repository

- **Structured eligibility criteria (285 trials).** Parsed from ClinicalTrials.gov protocols and decomposed into atomic features via GPT-4.1 (`data/{cancer_type}/structured_trial_data_*.csv`).
- **Trial metadata** (`data/trials_info_application_cleaned.json`). NCT IDs, enrollment, site count, recruitment duration, intervention category, and start date for the 285 study-cohort trials.
- **Clinician-annotated patient-trial pairs** (`data/annotation.json`). 289 patient-trial pairs labeled at the criterion level by two board-certified oncologists, with patient identifiers de-identified as sequential integers.
- **Sample pipeline outputs** (`results/`). Rule-level exclusion rates by cancer type and demographic subgroup, laboratory threshold summaries, and aggregated per-trial outputs.

### What is NOT included (and why)

- **All of Us patient-level data.** The NIH *All of Us* Research Program Data Use and Registration Agreement strictly prohibits redistribution of Registered Tier data, including `person_id` values, outside the Researcher Workbench. Qualified researchers may obtain access at https://www.researchallofus.org/register/ and then reproduce our patient-side pipeline using the notebooks in `allofus_patient_data/`.
- **Raw trial protocols.** Full reference trials from ClinicalTrials.gov are not redistributed; use `retrospective_analysis/fetch_clinicalTrial.py` to retrieve them from the ClinicalTrials.gov API v2.

### ClinicalTrials.gov

All clinical trial data come from https://clinicaltrials.gov/ (public, no license restrictions). The 1,937-trial historical reference database is retrieved on-the-fly via the API.

---

## Quick start

### Browse historical rule statistics without running anything

Visit the interactive web interface at https://aspect-trial.com/ to explore:

- Frequency rankings of eligibility criteria across 285 trials
- Average historical outcomes associated with each criterion
- Real-world exclusion rates by demographic subgroup
- Laboratory threshold sensitivity curves for 8 common parameters

### Reproduce analyses locally

```bash
# ── Retrospective ────────────────────────────────────────────────

# 1. Fetch historical trials from ClinicalTrials.gov
python retrospective_analysis/fetch_clinicalTrial.py -query "breast cancer" -size -1

# 2. Semantic-match extracted criteria across trials (SBERT + SciFive NLI)
python retrospective_analysis/retrive_relevent_ec_large_scale.py

# 3. Analyze rule frequency, enrollment productivity, and temporal trends
python retrospective_analysis/frequency_ranking.py
python retrospective_analysis/enrollment_analysis.py
python retrospective_analysis/longitudinal_analysis.py

# 4. Train transformer-based outcome predictors (requires TrialBench dataset)
python retrospective_analysis/bio_bert_train_classification.py
python retrospective_analysis/bio_bert_train_regression.py

# ── Prospective (All of Us side runs inside the Workbench) ───────

# 5. Inside the AoU Workbench: run the cancer-specific notebooks, then
python allofus_patient_data/generate_structure_patient_data.py

# 6. Structure trial eligibility criteria via GPT-4.1
python prospective_simulation/retrieve_structure_table_application.py

# 7. Run patient-trial matching
python prospective_simulation/perform_matching.py \
    --top_k 1 --evaluation nli --dataset lung_cancer \
    --matching_threshold 0.5 --partition 1 --gpu cuda:0

# 8. Benchmark against clinician annotations
python prospective_simulation/run_matching_benchmark.py \
    --top_k 1 --evaluation nli --dataset lung_cancer \
    --matching_threshold 0.5 --gpu cuda:0

# 9. Benchmark baseline LLMs
python prospective_simulation/run_benchmark.py

# 10. Compute benchmark metrics (Fig. 6)
python prospective_simulation/benchmark_evaluation.py

# ── Multi-objective optimization ─────────────────────────────────

# 11. Optimize a trial's eligibility criteria (example: ATLANTIS, NCT02566993)
python multi-objective_optimization/run_moga.py \
    -dataset lung_cancer -numGen 100 -popSize 100 \
    -seed 1 -min_active_rules 15 -fitness_type enrollment+adverse

# ── Website data ─────────────────────────────────────────────────

# 12. Aggregate statistics for the web interface
python website/prepare_website_data.py
```

## Reproducing paper figures

Each figure is regenerated from processed artifacts in `data/` and `results/`. Patient-dependent figures require All of Us access.

| Figure  | Content                                                          | Entry point                                             | Requires AoU |
|---------|------------------------------------------------------------------|---------------------------------------------------------|--------------|
| Fig. 1  | Framework overview                                               | schematic (no reproduction script)                      | No           |
| Fig. 2  | Rule-level exclusion by demographic subgroup                     | processed from `results/all_trials_rule_exclusion_rates.csv` | Yes (for regen) |
| Fig. 3  | Retrospective design patterns (top rules, EPSM, temporal trends) | `frequency_ranking.py`, `enrollment_analysis.py`, `longitudinal_analysis.py` | No |
| Fig. 4  | Laboratory threshold sensitivity                                 | `frequency_ranking_numerical.py` + `results/full_rule_summary_nli_check_0.6_numerical.json` | Partial |
| Fig. 5  | ATLANTIS (NCT02566993) case study + MOGA                         | `multi-objective_optimization/run_moga.py`              | Partial      |
| Fig. 6  | Patient–trial matching benchmark                                 | `prospective_simulation/benchmark_evaluation.py`        | Yes          |
| Fig. C1 | Lorenz curve of exclusion burden                                 | computed from `results/all_trials_rule_exclusion_rates.csv` | No     |
| Fig. C4 | Outcome-prediction calibration                                   | `retrospective_analysis/bio_bert_inference.py`          | No           |

---

## Ethics & data governance

- **ClinicalTrials.gov** data are publicly available and carry no use restrictions.
- **All of Us Research Program** data were accessed under Registered Tier credentials (Dataset v7) via the official Researcher Workbench. All analyses were performed inside the Workbench; no patient-level records leave that environment.
- Patient identifiers in `data/annotation.json` have been replaced with sequential integers; the original `person_id` mapping is retained privately and is not part of this release.
- This work constitutes secondary analysis of de-identified data and did not require separate IRB review at the contributing institutions.
- Researchers who wish to reproduce the prospective-simulation component must obtain their own AoU access through https://www.researchallofus.org/register/.

---

## Contact

**Corresponding author:** Ruishan Liu — Department of Computer Science, USC Viterbi
**Lead author:** Konghao Zhao — Department of Computer Science, USC Viterbi

Issues, feature requests, and questions: please open a GitHub issue.

---

## License

MIT License — see [LICENSE](LICENSE) for full terms.

---

## Acknowledgements

We thank the *All of Us* Research Program and its participants for making the Registered Tier Dataset (v7) available through the Researcher Workbench, without which this research would not have been possible.
