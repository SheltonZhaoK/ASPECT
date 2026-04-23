# +
def patient_prompt_reference(reference):
    patient_prompt = f"""You are given a paragraph describing a patient. Extract all identifiable features of the patient, and generate a list of dictionaries.
Each dictionary should contain three keys: 'name', 'description', and 'value'.

Instructions:
1. Use the provided reference to identify specific features. For each feature:
   - The 'name' is the reference entry excluding words in parenthesis
   - The 'description' is a brief description of 'name' with the unit concatenated. The 'description' should be semantically meaningful based on context. 
   - The 'value' is the specific value patient has for the 'name', ensuring it is numerical, categorical, a list, or a description if necessary.
   - Convert unit of patient 'value' as necessary based on the reference's unit.
   - If a feature is missing in the patient description, ignore it.

2. Dynamically identify additional relevant patient features not specified in the reference. For each feature:
   - The 'name' is the abbreviations or short phrases for feature name
   - The 'description' is a brief description of 'name' with the unit concatenated. The 'description' should be semantically meaningful based on context.
   - The 'value' is the specific value patient has for the 'name', ensuring it is numerical, categorical, a list, or a description if necessary.
   - Ensure that each feature is semantically meaningful.

3. Output Format:x
   - For each feature, create a dictionary with three keys: 'name', 'description', and 'value'.
   - Use consistent descriptions for repeated features.

Here is the reference: {reference}

Example Input:
'A 90-year-old Asian woman is taken to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life, she currently takes no medications, she has a body temperature of 36.7 C, she has a hand tremor confirmed by CT ...'

Example Output:
"[
    {{"name": "age", "description": "age of the patient", "value": "90"}},
    {{"name": "ethnicity", "description": "race of the patient", "value": "Asian"}},
    {{"name": "gender", "description": "gender of the patient", "value": "female"}},
    {{"name": "temperature", "description": "body temperature (°C)", "value": "36.7"}},
    ...
    {{"name": "ER administration", "description": "administered to an emergency room", "value": "yes"}},
    {{"name": "chest pain", "description": "episodic pressing/burning anterior chest pain", "value": "yes, began two days ago"}},
    {{"name": "medications", "description": "current medications", "value": "no"}},
    {{"name": "hand tremor", "description": "a hand tremor", "value": "yes, confirmed by CT"}}
    ...
]"

Here is the given text. Based on this, extract and format the features according to the above guidelines:
"""
    return patient_prompt

def patient_prompt_oct25():
    patient_prompt = """
You are given a paragraph describing a patient, extract all possible features of this patient. Generate a list of dictionaries, \
where each dictionary contains three keys: feature name, feature description and value.
Instructions:
1. Identify all features of the given patient, use abbreviation or short phrases as feature name, record the description and value of that feature.
2. Make sure the features are semantically meaningful, such that it could be later used for clinical trial matching examination. 
3. Format output: For each feature, create a dictionary with three keys: "name", "description", "value" and fill in the values. For the same feature name, use the same description.
Example input:
'A 90-year-old Asian woman is taken to the ER with episodic pressing/burning anterior chest pain that began two days \
earlier for the first time in her life, she currently takes no medications, she has a hand tremor confirmed by ct ... '
Example Output Format:
"[
    {"name": "age", "description": "age of the patient", "value": "90 years old"},
    {"name": "race", "description": "race of the patient", "value": "asian"},
    {"name": "gender", "description": "gender of the patient", "value": "female"},
    {"name": "ER administration", "description": "administered to an emergency room", "value": "yes"},
    {"name": "chest pain", "description": "episodic pressing/burning anterior chest pain", "value": "yes and began two days"},
    {"name": "medications", "description": "current medications", "value": "no"},
    {"name": "hand tremor", "description": "a hand tremor", "value": "yes and confirmed by ct"}
    ...
]"

Here is the given text. Based on this, extract and format the features according to the above guidelines:
"""
    return patient_prompt


def trial_prompt():
    trial_prompt = """
You are given a paragraph of clinical trial description, extract and organize the inclusion and exclusion criteria into a dictionary. Criteria are keys, with 'yes' for inclusion and 'no' for exclusion.
Instructions:
1. Identify Criteria Sections: Locate the sections labeled as "inclusion criteria" and "exclusion criteria" in the input text. These sections might be clearly labeled or implicitly labeled. 
2. Extract Criteria: For 'inclusion criteria', list each criterion after the phrase. For 'exclusion Criteria', do the same.
3. Format Output: Create a dictionary with each criterion as a key and assign 'yes' for inclusion and 'no' for exclusion.
4. Make sure the criteria are down to the atom level
Example Input:
'inclusion Criteria: the patient needs to have Congenital Adrenal Hyperplasia (CAH) and ... exclusion Criteria: the patient has history of cardiovascular disease and ...'
Example Output Format:
"[
    {"Congenital Adrenal Hyperplasia (CAH)": "yes"},
    {"history of cardiovascular disease": "no"}
]"
Steps:
- Split the input string to separate inclusion and exclusion criteria.
- Extract and list criteria from each section.
- Populate a dictionary with criteria as keys and their status ('yes'/'no') as values.
Here is the given text. Based on this, extract and format the criteria according to the above guidelines:
"""
    return trial_prompt

def inclusion_trial_prompt():
    trial_prompt = """
You are given a paragraph of a clinical trial description. Your task is to extract and organize the inclusion criteria into a dictionary, with criteria as keys and 'yes' as their values.
Instructions:
Extract Criteria: Identify and list each criterion under the "inclusion criteria" section of the trial description.
Format Output: Create a dictionary where each criterion is a key, and assign 'yes' as the value for each entry.
Ensure Semantic and Biomedical Relevance: Make sure that each extracted criterion has clear semantic and biomedical meaning for later comparison.
Add Missing Information: If a criterion lacks sufficient detail, enrich it with additional relevant information.
Preserve Original Meaning: Do not alter or negate the meaning of the criterion—extract it exactly as stated.
Do Not Separate Related Criteria: If multiple criteria share a common feature or concept, group them together and ensure that related information is not fragmented and remains logically connected.
Example Input:
"inclusion Criteria: the patient needs to have Congenital Adrenal Hyperplasia (CAH), should not be older than 50, and does not have other diseases, weights > 130lbs in males > 15years, weights < 120lbs in males < 15, weights < 110lbs in females < 20."
Example Output Format:
"[
    {"Congenital Adrenal Hyperplasia (CAH)": "yes"},
    {"Not older than 50": "yes"},
    {"No other diseases than Congenital Adrenal Hyperplasia (CAH)": "yes"},
    {"weights > 130lbs in males > 15years or weights < 120lbs in males < 15years or weights < 110lbs in females < 20years": "yes"}
]"
Steps:
- Understand each criterion in the trial description.
- Extract and list each criterion while ensuring it has semantic and biomedical meaning.
- Populate a dictionary with criteria as keys and 'yes' as values.
Input Text:
Use the given text to extract and format the criteria according to the guidelines above. Don't generate the code, instead generate the output directly.
"""
    return trial_prompt

def exclusion_trial_prompt():
    trial_prompt = """
You are provided with a paragraph from a clinical trial description. Your task is to extract and organize the exclusion criteria into a dictionary, where each criterion is a key, and the value is 'no.'
Instructions:
Extract Criteria: Identify and list each criterion under the "exclusion criteria" section of the trial description.
Format Output: Create a dictionary where each criterion is a key, and assign 'no' as the value for each entry.
Ensure Semantic and Biomedical Relevance: Ensure that each extracted criterion retains clear semantic and biomedical meaning for accurate comparison later.
Add Missing Information: If any criterion lacks sufficient detail, enrich it with additional relevant information.
Preserve Original Meaning: Do not alter or negate the meaning of the criterion (extract it exactly as stated).
Do Not Separate Related Criteria: If multiple criteria share a common feature or concept, group them together and ensure that related information is not fragmented and remains logically connected.
Example Input:
"exclusion Criteria: the patient is pregnent, admitted to ICU more than 10 days, has a hand tremor, weights > 130lbs in males > 15years, weights < 120lbs in males < 15, weights < 110lbs in females < 20.."
Example Output Format:
"[
    {"pregnent": "no"},
    {"admitted to ICU more than 10 days": "no"},
    {"hand tremor": "no"},
    {"weights > 130lbs in males > 15years or weights < 120lbs in males < 15years or weights < 110lbs in females < 20years": "no"}
]"
Steps:
- Understand each criterion in the trial description.
- Extract and list each criterion while ensuring it has semantic and biomedical meaning.
- Populate a dictionary with criteria as keys and 'no' as values.
Input Text:
Use the given text to extract and format the criteria according to the guidelines above. Don't generate the code, instead generate the output directly.
"""
    return trial_prompt

def feature_relevance_prompt(patient_feature_list, trial_feature):
    prompt = (
    "You are tasked with determining if two descriptions, one from a patient and one from a clinical trial are relevant, that is whether they "
    "refer to the same concept or context. Respond with 'yes' if they are referring to the same general concept (e.g., both talking about age, symptoms, condition, disease, health score, and etc.), "
    "or 'irrelevant' if they are not (e.g., one talks about age and the other talks about symptoms or one talks about tremor and the other talks about fever). "
    f"The patient feature is '{'.'.join(patient_feature_list)}', and the trial feature is '{trial_feature}'. "
    "Format your response as 'Final Answer: <your_answer>'.")
    return prompt

def values_matching_prompt(patient_feature, trial_feature, patient_value):
    prompt = f"Consider a patient who has a characteristic labeled as '{patient_feature}' with a value of '{patient_value}'. Evaluate if this patient qualifies for a specific trial criterion labeled as '{trial_feature}.' You should answer with 'yes' if the patient's condition aligns with the trial requirement, 'no' if it contradicts, or 'N/A' if there is not enough information. Please format your answer as 'Final Answer: <your_answer>'."
    return prompt

def values_matching_prompt_v2(patient_feature, trial_feature, patient_value):
    prompt = f"Consider a patient who has a characteristic labeled as '{patient_feature}' with a value of '{patient_value}'. Evaluate if this patient qualifies for a specific trial criterion labeled as '{trial_feature}.' You should answer with 'yes' if the patient's condition aligns with the trial requirement, 'no' if it contradicts, or 'N/A' if there is not enough information. Here are several notes that aid your reasoning: 1. consider the semantic meaning between two context instead of boolean logic and simple comparison; 2. For numerical comparison, if two numbers are not in the same unit of measurement make them the same; 3. For ethnicity, one race might include a group of people. Please format your answer as 'Final Answer: <your_answer>'."
    return prompt

def criterion_description_prompt(patient, trial, value):
    prompt = f"""
Consider a patient who has a description '{patient}'. Your task is to determine if this patient qualifies for a specific trial criterion '{trial}' with a value of '{value}'.

**Follow these steps carefully in your reasoning:**
1. **Identify Relevance**:
   - Determine if the patient description is related to the trial criterion in a biomedical or semantic context.
   - If nothing is related, respond with 'Final Answer: irrelevant' without further analysis.

2. **Analyze Similarity**:
   - If relevant, compare the semantic and biomedical meaning between the patient description and the trial criterion.
   - For numerical features (e.g., age or lab results), ensure units of measurement are consistent before making comparisons.

3. **Evaluate Eligibility**:
   - If the patient meets the trial criterion, respond with 'Final Answer: yes'.
   - Only respond with 'Final Answer: no' if there is clear evidence in patient that does not meet the trial criterion. Otherwise, respond with 'Final Answer: N/A'
   - Also, if the information provided is insufficient to make a decision, respond with 'Final Answer: N/A'.

**Structure your response as follows:**
- Clearly outline the relevance, analysis, and conclusion.
- Provide your final answer in this format: 'Final Answer: <your_answer>'.

Evaluate the patient's eligibility based on the above reasoning steps and provide your structured final answer.
"""
    return prompt

def values_matching_prompt_cot(patient_feature_list, trial_feature, patient_value_list, trial_value):
    patient_info = ", ".join(f"{feature}: {value}" for feature, value in zip(patient_feature_list, patient_value_list))
    prompt = f"""
Consider a patient with the following characteristics: {patient_info}.\
Your task is to determine if this patient qualifies for a specific trial criterion labeled as '{trial_feature}' with a value of '{trial_value}'.

**Follow these steps carefully in your reasoning:**
1. **Identify Relevance**:
   - Determine if the patient characteristic and the trial criterion are related in a biomedical or semantic context.
   - If they are not directly related (e.g., comparing age to blood type), respond with 'Final Answer: irrelevant' without further analysis.

2. **Analysis and Evaluate Eligibility**:
   - Infer necessary information from patients from your common sense, patient's symptoms, medical knowledge, and condition prevalence to aid with decision when possible. 
   - For numerical features (e.g., age or lab results), ensure units of measurement are consistent before making comparisons.
   - If the patient meets the trial criterion, respond with 'Final Answer: yes'.
   - If the patient does not meet the trial criterion, respond with 'Final Answer: no'.
   
3. **Handle Uncertainty**:
   - If the information provided is insufficient to make a decision, respond with 'Final Answer: N/A'.

Clearly outline the relevance, analysis, and conclusion. Provide your final answer in this format: 'Final Answer: <your_answer>'.\
Evaluate the patient's eligibility based on the above reasoning steps and provide your structured final answer.
"""
    return prompt


def text_matching_prompt(patient, trial):
    prompt = f"Consider a patient described as '{patient}' and a clinical trial with eligibility criteria outlined as '{trial}'. Determine whether the patient qualifies for the trial.\
Conclude with 'yes' if the patient is eligible for the trial, or 'no' if the patient is not eligible.\
Please structure your final response in the following format:\
'Final Answer: <yes or no>'"
    return prompt

# # Example Input:
# # 'A 90-year-old Asian woman is taken to the ER, she currently takes no medications, she denies hypertension and Diabetes, she has a body temperature of 36.7°C and blood pressure of 199/108, she has a hand tremor confirmed by CT, her sodium (mEq/l) level is 137, her LDL-C level (mg/dL) level is 500....'

# # Example Output:
# # "[
# #     {{"name": "age (year)", "value": "90"}},
# #     {{"name": "gender", "value": "female"}},
# #     {{"name": "ethnidef patient_prompt_step2(patient_features, num_reference, cat_reference, dis_reference, sym_reference):
#     patient_prompt = f"""
# Analyze the provided patient description, identify their variable type, and structure them into a list of dictionaries with three keys: 'type' the variable type,\
# 'name' a concise description that includes units if applicable, and 'value' the corresponding value.

# First, analyze the structure of the given patient description, identify whether it has complex structure or modifiers.\
# If the description has complex structure or modifiers which means it cannot be decomposed into simple feature without losing information, \
# for example it may contain severity levels, durations, pain levels, lab test or detailed descriptions of symptoms or diseases, convert the description to a descriptive variable.\
# Use "descriptive" as the value of 'type', "n/a" as the value of 'value', and retain the original feature description as the value of 'name'.\

# Second, if the description could be decomposed, classify it as either numerical, demographic, disease, or symptom variable.\
# Check whether it is numerical variable first, if it is, include the unit in parentheses in the 'name'. For numerical variable, split blood pressure into systolic and diastolic blood pressure.\
# Convert all age to years and body temperature to celsius.\
# Then check whether it is demographic variable. The 'name' and 'value' of demographic variables are predefined in the reference {cat_reference}. Only use what is available in the reference.\
# If the feature does not appear in reference, it is either disease variable or symptom variable. Only use "diagnosed" or "not diagnosed" for disease variable as the value\
# and "diagnosed" or "not diagnosed" for symptom variables as the value. 

# ### Example Input:
# [
#     {{"description": "the patient is 90 years old"}},
#     {{"description": "the patient is a female"}},
#     {{"description": "the patient is Asian"}},
#     {{"description": "the patient has a body temperature (°C) of 36.7"}},
#     {{"description": "the patient has a blood pressure (mm Hg) of 199/108"}},
#     {{"description": "the patient does not have hypertension"}},
#     {{"description": "the patient has a headache hypertension"}},
#     {{"description": "the patient has a hand tremor confirmed by CT"}}
#     ...
# ]
# ### Example Output:
# [
#     {{"type": "numerical", "name": "age (years)", "value": "90"}}
#     {{"type": "demographic", "name": "gender", "value": "female"}}
#     {{"type": "demographic", "name": "ethnicity", "value": "asian"}}
#     {{"type": "numerical", "name": " body temperature (°C)", "value": "36.7"}}
#     {{"type": "numerical", "name": "systolic blood pressure (mm Hg)", "value": "199"}}
#     {{"type": "numerical", "name": "diastolic blood pressure (mm Hg)", "value": "108"}}
#     {{"type": "disease", "name": "hypertension", "value": "not diagnosed"}}
#     {{"type": "symptom", "name": "headache", "value": "present"}}
#     {{"type": "descriptive", "name": "hand tremor confirmed by CT", "value": ""n/a""}}
#     ...
# ]

# Provide only the formatted output (no Python code or commentary).\
# Convert the provided patient description strictly according to the guidelines without adding, modifying, or inferring any new information.\
# Return the processed features as a structured list of dictionaries: {patient_features}
# """
#     return patient_prompt

def patient_prompt(num_reference, cat_reference):
    patient_prompt = f"""
You are given a paragraph describing a patient. Extract all identifiable features and generate a structured list of dictionaries.
Each dictionary should contain 'name' a concise description of the feature and 'value' the feature's corresponding value.

### Feature Extraction Guidelines:
1. **Numerical Attributes**: 
    For similar feature, use the same feature names and convert the value to the same unit as in the reference: {num_reference}. 
    Include unit closed by parenthesis in the **'name'** to obtain similar format of feature name as in the reference.
    Split blood pressure to systolic blood pressure and diastolic blood pressure. 
        - Example: 
            - input: 'patient has a blood pressure of 155/98' 
            - output: {{"name": "systolic blood pressure (mmhg)", "value": "155"}}
            - output: {{"name": "diastolic blood pressure (mmhg)", "value": "98"}}
            
    For non-specified numerical features in the reference, just output it. 
        - Example: {{"name": "age (years)", "value": "45"}}
        - Example: {{"name": "blood fat (mg/dl)", "value": "180"}}

2. **Categorical Attributes**: Match features to the specified categories in the reference: {cat_reference}. For similar semantic context, use the feature name in the reference. For example, woman should be converted to female in the reference.

3. **Disease Features**: Record as only 'diagnosed' or 'not diagnosed'.
    - Example:
        - {{"name": "hypertension", "value": "diagnosed"}}
        - {{"name": "diabetes", "value": "not diagnosed"}}

4. **Symptom Features**: Record as only 'present' or 'not present'.
    - Example:
        - {{"name": "fever", "value": "present"}}
        - {{"name": "anxiety", "value": "present"}}

5. **Discriptive Features**: 
        - Use descriptive and semantically meaningful description for non-numerical, categorical, disease, or sympton-related features as 'name'.
        - Use the word 'descriptive' as values

6. **List Features**: If exitst, clearly list multiple values.
        - Example: {{"name": "drug allergies", "value": ["penicillin", "aspirin"]}}

3. **Notes**:
    - Keep 'name' of numerical and categorical features concise
    - Ensure semantic context for discriptive features

### Output Format:
Provide the extracted features as a list of dictionaries. Do not write any Python code or commentary, only the extracted features.

### Patient Description:
Extract and format features from the following patient description according to the above guidelines:
"""

    return patient_prompt


# def patient_prompt_step1():
#     patient_prompt = f"""
# You are given a paragraph describing a patient. Extract all identifiable features and structure them as a list of dictionaries.
# Each dictionary should have a single key, 'description' with its value containing a complete and meaningful description of a patient feature.

# ### Feature Extraction Guidelines:
# - Extract each feature as a descriptive sentence.
# - For numerical features, include their units.
# - Ensure all descriptions are semantically meaningful and contextually accurate.
# - State the sentence with 'the patient'

# ### Example Input:
# "A 90-year-old Asian woman is taken to the ER, she currently takes no medications, she has a body temperature of 36.7°C and blood pressure of 199/108, she denies hypertension, she has a hand tremor confirmed by CT, her sodium (mEq/l) level is 137, her LDL-C level (mg/dL) level is 500...."

# ### Example Output:
# [
#     {{"description": "the patient is 90 years old"}},
#     {{"description": "the patient is a female"}},
#     {{"description": "the patient is Asian"}},
#     {{"description": "the patient has a body temperature (°C) of 36.7"}},
#     {{"description": "the patient has a blood pressure (mm Hg) of 199/108"}},
#     {{"description": "the patient has a sodium (mEq/l) level of 137"}},
#     {{"description": "the patient has an LDL-C (mg/dL) level of 500"}},
#     {{"description": "the patient does not have hypertension"}},
#     {{"description": "the patient is admitted to the emergency room"}},
#     {{"description": "the patient does not currently take any medications"}},
#     {{"description": "the patient has a hand tremor confirmed by CT"}}
#     ...
# ]

# ### Notes:
# - Use concise, consistent phrasing for all features.
# - Ensure descriptions are semantically accurate and contextually clear.

# ### Output Format:
# Return the extracted features as a list of dictionaries, formatted as shown above. Do not include any Python code or commentary.

# ### Patient Description:
# Extract and format features from the following patient description:
# """
#     return patient_prompt

# def patient_prompt_step2(patient_features, num_reference, cat_reference, dis_reference, sym_reference):
#     patient_prompt = f"""
# Analyze the provided patient features, identify their variable type, and structure them into a list of dictionaries with:
# - **'type'**: The variable type.
# - **'name'**: A concise description (include units if applicable).
# - **'value'**: The corresponding value.

# ### Instructions:
# 1. **Descriptive Variables** (for complex descriptions that cannot be decomposed):
#     - If the value of the feature has complex structure or cannot be decomposed without losing information, classify as descriptive.
#     - These variables typically include information such as severity levels, durations, pain levels, or detailed descriptions of symptoms or diseases.
#     - Use "n/a" as the **'value'**, retain the original feature description as the **'name'**.
#         - Example:
#             - Input: "Patient has a mild dyspnea three days ago"
#             - Output: {{"type": "descriptive", "name": "Patient has a mild dyspnea three days ago", "value": "n/a"}}

# 2. **Numerical Variables**:
#     - Refer to {num_reference} for feature names and units. Match naming convention and convert values to the same units.
#     - Include the unit in parentheses in the **'name'** field to match the reference format.
#     - Convert pulse rate to heart rate
#     - For blood pressure, split into systolic and diastolic components.
#         - Example:
#             - Input: "Patient has a blood pressure of 155/98"
#             - Output: 
#                 - {{"type": "numerical", "name": "systolic blood pressure (mmHg)", "value": "155"}}
#                 - {{"type": "numerical", "name": "diastolic blood pressure (mmHg)", "value": "98"}}
#     - Retain numerical features not found in the reference with an appropriate name.
#         - Example:
#             - {{"type": "numerical", "name": "blood fat (mg/dL)", "value": "180"}}

# 3. **Categorical Variables**:
#     - If the criterion does not have any qualitative modifiers and could be decomposed to a predefined categorical variable without losing information as in {cat_reference}, extract the categorical variable and match naming convention to categories in the reference.
#     - Use consistent naming and category from the reference.
#         - Example:
#             - Input: "woman"
#             - Output: {{"type": "categorical", "name": "gender", "value": "female"}}

# 4. **Disease Variables**:
#     - If the criterion does not have any qualitative modifiers and could be decomposed to a single disease variable without losing information, match naming convention to disease in {dis_reference}; retain others not in the reference with an appropriate name.
#     - Only use "diagnosed" or "not diagnosed" as the value, otherwise make the variable descriptive.
#         - Example:
#             - {{"type": "disease", "name": "hypertension", "value": "diagnosed"}}

# 5. **Symptom Variables**:
#     - If the criterion does not have any qualitative modifiers and could be decomposed to a single symptom variable without losing information, match naming convention to symptom in {sym_reference}; retain others not in the reference with an appropriate name.
#     - Only use "present" or "not present" as the value, otherwise make the variable descriptive.
#         - Example:
#             - {{"type": "symptom", "name": "fever", "value": "present"}}

# 6. **List Variables**:
#     - Clearly list all items in the **'value'** field.
#         - Example:
#             - Input: "Patient is allergic to penicillin and aspirin"
#             - Output: {{"type": "list", "name": "drug allergies", "value": ["penicillin", "aspirin"]}}

# ### Output Format:
# Return the processed features as a structured list of dictionaries. Provide only the formatted output—no Python code or commentary.

# ### Patient Description:
# Convert the provided patient description strictly according to the formatting guidelines without adding, modifying, or inferring any new information, the number of variable outputed should be equal to the number of description: {patient_features}
# """
#     return patient_prompt

# def trial_prompt_step1(trial_criteria):
#     trial_prompt = f"""
# You are given a paragraph containing a clinical trial eligibility description. Your task is to extract and organize the criteria into a dictionary format.
# Each dictionary should have a single key, 'description' with its value containing a complete and meaningful description of the trial criterion.

# ### Feature Extraction Guidelines:
# - Extract each feature as a descriptive sentence.
# - For numerical features, include their units.
# - Ensure all descriptions are semantically meaningful and contextually accurate.
# - State the sentence with 'the patient'

# ### Example Input:
# "Criterion: older than 18, younger than 60, pregnancy, lactation, insulin-dependent diabetes with dysautonomia, agitation, central or peripheral neurological disease"

# ### Example Output:
# [   
#     {{"description": "the patient is aged between 18 and 60 years old"}},
#     {{"description": "the patient is pregnant"}},
#     {{"description": "the patient is lactating"}},
#     {{"description": "the patient has insulin-dependent diabetes with dysautonomia"}},
#     {{"description": "the patient has central or peripheral neurological disease"}},
#     {{"description": "the patient has agitation"}}
# ]

# Below is the eligibility description. Extract and format the features according to the above guidelines. Don't generate the python code, extract and generate the outputs directly:
# {trial_criteria}
# """
#     return trial_prompt

# def trial_prompt_step2(trial_features, summary=None, num_reference=None, cat_reference=None, dis_reference=None, sym_reference=None):
#     summary_text = f"### Step 3: Highlight Important Criterion\nBased on the trial summary: {summary}, identify the most important criterion. Mark its **'type'** as 'important'.\n" if summary is not None else ""
    
#     trial_prompt = f"""
# Analyze the provided trial criterion, convert each criterion to a specific variable, and structure them into a list of dictionaries, each containing:
# - **'type'**: The variable type.
# - **'name'**: A concise description (include units if applicable).
# - **'value'**: The corresponding value.

# ### Instructions
# 1. **Descriptive Variables**:
#     - If the value of the feature has complex structure or modifiers or cannot be decomposed without losing information, classify as descriptive.
#     - These variables typically include information such as severity levels, durations, pain levels, or detailed descriptions of symptoms or diseases.
#     - Use "n/a" as the **'value'**, retain the original feature description as the **'name'**.
#         - Example:
#             - Input: "Patient has a mild dyspnea three days ago"
#             - Output: {{"type": "descriptive", "name": "Patient has a mild dyspnea three days ago", "value": "n/a"}}

# 2. **Numerical Variables**:
#     - If the criterion could be decomposed to a single numercial variable without losing information, refer to {num_reference} for criterion names and units. Match naming convention and convert values to the same units.
#     - Include the unit in parentheses in the **'name'** field to match the reference format.
#     - Convert pulse rate to heart rate.
#     - Split the numerical variables into two variables (min and max), for example the phrase 'greater than' represents min value and vice versa.
#     - For blood pressure, split into systolic and diastolic components.
#         - Example:
#             - Input: "Patient has a blood pressure higher than 155/98"
#             - Output: 
#                 - {{"type": "numerical", "name": "max systolic blood pressure (mmHg)", "value": "n/a"}}
#                 - {{"type": "numerical", "name": "max diastolic blood pressure (mmHg)", "value": "n/a"}}
#                 - {{"type": "numerical", "name": "min systolic blood pressure (mmHg)", "value": "155"}}
#                 - {{"type": "numerical", "name": "min diastolic blood pressure (mmHg)", "value": "98"}}
#     - Retain numerical features not found in the reference with an appropriate name.

# 3. **Categorical Variables**:
#     - If the criterion does not have any qualitative modifiers and could be decomposed to a predefined categorical variable without losing information as in {cat_reference}, extract the categorical variable and match naming convention to categories in the reference.
#     - Use consistent naming and category from the reference.
#         - Example:
#             - Input: "the patient is pregnant"
#             - Output: {{"type": "categorical", "name": "pregnancy status", "value": "pregnant"}}

# 4. **Disease Variables**:
#     - If the criterion does not have any qualitative modifiers and could be decomposed to a single disease variable without losing information, match naming convention to disease in {dis_reference}; retain others not in the reference with an appropriate name.
#     - Only use "diagnosed" or "not diagnosed" as the value, otherwise make the variable descriptive.
#         - Example:
#             - {{"type": "disease", "name": "hypertension", "value": "diagnosed"}}

# 5. **Symptom Variables**:
#     - If the criterion does not have any qualitative modifiers and could be decomposed to a single symptom variable without losing information, match naming convention to symptom in {sym_reference}; retain others not in the reference with an appropriate name.
#     - Only use "present" or "not present" as the value, otherwise make the variable descriptive.
#         - Example:
#             - {{"type": "symptom", "name": "fever", "value": "present"}}

# 6. **List Variables**:
#     - Clearly list all items in the **'value'** field.
#         - Example:
#             - Input: "Patient is allergic to penicillin and aspirin"
#             - Output: {{"type": "list", "name": "drug allergies", "value": ["penicillin", "aspirin"]}}

# {summary_text}
# ### Output Format:
# Return the processed features as a structured list of dictionaries. Provide only the formatted output—no Python code or commentary.

# ### Trial Description:
# Convert the provided trial criteria strictly according to the formatting guidelines without adding, modifying, or inferring any new information. The number of variable outputed should be equal to the number of description : {trial_features}
# """
#     return trial_prompt


def patient_prompt_step1():
    patient_prompt = f"""
You are given a paragraph describing a patient. Extract all identifiable features and structure them as a list of dictionaries.\
Each dictionary should have a single key called 'description' with its value containing a complete and meaningful description of a patient feature. \
You need to extract each feature as a descriptive sentence. For numerical features, include their units. Ensure all descriptions are semantically meaningful and contextually accurate\
and state the sentence with 'the patient'.
### Example Input:
"A 90-year-old Asian woman is taken to the ER, she has a body temperature of 36.7°C and blood pressure of 199/108, she denies hypertension, \
she has a hand tremor confirmed by CT, her LDL-C level (mg/dL) level is 500...."
### Example Output:
[
    {{"description": "the patient is 90 years old"}},
    {{"description": "the patient is a female"}},
    {{"description": "the patient is Asian"}},
    {{"description": "the patient has a body temperature (°C) of 36.7"}},
    {{"description": "the patient has a blood pressure (mm Hg) of 199/108"}},
    {{"description": "the patient has an LDL-C (mg/dL) level of 500"}},
    {{"description": "the patient does not have hypertension"}},
    {{"description": "the patient is admitted to the emergency room"}},
    {{"description": "the patient has a hand tremor confirmed by CT"}}
    ...
]
Do not include any Python code or commentary. Extract and format features from the following patient description:
"""
    return patient_prompt

def patient_prompt_step2(patient_features, num_reference, cat_reference, dis_reference, sym_reference):
    patient_prompt = f"""
Analyze the provided patient description, identify their variable type, and structure them into a list of dictionaries with three keys: 'type' the variable type,\
'name' a concise description that includes units if applicable, and 'value' the corresponding value.

First, analyze the structure of the given patient description, identify whether it has complex structure or modifiers.\
If the description has complex structure or modifiers which means it cannot be decomposed into simple feature without losing information, \
for example it may contain severity levels, durations, pain levels, lab test or detailed descriptions of symptoms or diseases, convert the description to a descriptive variable.\
Use "descriptive" as the value of 'type', "n/a" as the value of 'value', and retain the original feature description as the value of 'name'.\

Second, if the description could be decomposed, classify it as either numerical, demographic, or condition variable.\
Check whether it is numerical variable first, if it is, include the unit in parentheses in the 'name'. For numerical variable, split blood pressure into systolic and diastolic blood pressure and \
convert all age to years and body temperature to celsius.\
Then, refer to the demographic reference: {cat_reference}, where the 'name' and 'value' of demographic variables are predefined.\
If a patient's feature and value could be reduced or found in the reference, it is a demographic variable and extract accordingly.\
If the feature does not appear in the demographic reference, it is a condition variable which represents either disease or symptom.\
Use "present" or "not present" as the value for condition variable.\
For any other cases not covered, make it descriptive variable.

### Example Input:
[
    {{"description": "the patient is 90 years old"}},
    {{"description": "the patient is a female"}},
    {{"description": "the patient is currently smoking"}},
    {{"description": "the patient has a body temperature (°C) of 36.7"}},
    {{"description": "the patient has a blood pressure (mm Hg) of 199/108"}},
    {{"description": "the patient does not have hypertension"}},
    {{"description": "the patient has a headache"}},
    {{"description": "the patient has a hand tremor confirmed by CT"}}
    ...
]
### Example Output:
[
    {{"type": "numerical", "name": "age (years)", "value": "90"}}
    {{"type": "demographic", "name": "gender", "value": "female"}}
    {{"type": "demographic", "name": "smoking status", "value": "yes"}}
    {{"type": "numerical", "name": " body temperature (°C)", "value": "36.7"}}
    {{"type": "numerical", "name": "systolic blood pressure (mm Hg)", "value": "199"}}
    {{"type": "numerical", "name": "diastolic blood pressure (mm Hg)", "value": "108"}}
    {{"type": "condition", "name": "hypertension", "value": "not present"}}
    {{"type": "condition", "name": "headache", "value": "present"}}
    {{"type": "descriptive", "name": "hand tremor confirmed by CT", "value": ""n/a""}}
    ...
]

Provide only the formatted output (no Python code or commentary). Don't use hyphen or dash to connect two words. For example, you should generate african american instead of african-american.\
Convert the provided patient description strictly according to the guidelines without adding, modifying, or inferring any new information.\
Return the processed features as a structured list of dictionaries: {patient_features}
"""
    return patient_prompt

def trial_prompt_step1(trial_criteria):
    trial_prompt = f"""
You are given a paragraph containing a clinical trial eligibility description. Your task is to extract and organize the criteria into a dictionary format.\
Each dictionary should have a single key called 'description' with its value containing a complete and meaningful description of a patient feature. \
You need to extract each feature as a descriptive sentence. For numerical features, include their units. Ensure all descriptions are semantically meaningful and contextually accurate\
and state the sentence with 'the patient'.

### Example Input:
"Criterion: older than 18, younger than 60, pregnancy, insulin-dependent diabetes with dysautonomia, agitation, central or peripheral neurological disease"

### Example Output:
[   
    {{"description": "the patient is aged between 18 and 60 years old"}},
    {{"description": "the patient is pregnant"}},
    {{"description": "the patient has insulin-dependent diabetes with dysautonomia"}},
    {{"description": "the patient has central or peripheral neurological disease"}},
    {{"description": "the patient has agitation"}}
]

Below is the eligibility description. Extract and format the features according to the above guidelines. Don't generate the python code, extract and generate the outputs directly:
{trial_criteria}
"""
    return trial_prompt

def trial_prompt_step2(trial_features, summary=None, num_reference=None, cat_reference=None, dis_reference=None, sym_reference=None):    
    trial_prompt = f"""
Analyze the provided trial description, identify their variable type, and structure them into a list of dictionaries with three keys: 'type' the variable type,\
'name' a concise description that includes units if applicable, and 'value' the corresponding value.

First, analyze the structure of the given trial description, identify whether it has complex structure or modifiers.\
If the description has complex structure or modifiers which means it cannot be decomposed into simple feature without losing information, \
for example it may contain severity levels, lab test, durations, pain levels, or detailed descriptions of symptoms or diseases, convert the description to a descriptive variable.\
Use "descriptive" as the value of 'type', "n/a" as the value of 'value', and retain the original feature description as the value of 'name'.\

Second, if the description could be decomposed, classify it as either numerical, demographic, disease, or symptom variable.\
Check whether it is numerical variable first, if it is, include the unit in parentheses in the 'name'. If this numerical feature specifies a minimum threshold, \
set it as 'min feature name (unit)' and vice versa'. If it specifies a range, break it into a min and a max feature. \
Also, split blood pressure into systolic and diastolic blood pressure, \
and convert all age to years and body temperature to celsius.\
Then, refer to the demographic reference: {cat_reference}, where the 'name' and 'value' of demographic variables are predefined.\
If a trial's feature and value could be reduced or found in the reference, it is a demographic variable and extract accordingly.\
If the feature does not appear in the demographic reference, it is a condition variable which represents either disease or symptom.\
Ensure the condition in its most precise medical form rather than descriptive.\
Use "present" or "not present" as the value for condition variable.\
For any other cases not covered, make it descriptive variable.

### Example Input:
[
    {{"description": "the patient can be a female or male"}},
    {{"description": "the patient is aged between 18 and 60 years old"}},
    {{"description": "the patient has granulocyte count ≥ 1200/μl"}},
    {{"description": "the patient is pregnant"}},
    {{"description": "the patient has hypertension"}},
    {{"description": "the patient has central or peripheral neurological disease"}},
    {{"description": "the patient does not has agitation"}}
    ...
]
### Example Output:
[
    {{"type": "demographic", "name": "gender", "value": "both"}}
    {{"type": "numerical", "name": "max age (years)", "value": "60"}}
    {{"type": "numerical", "name": "min age (years)", "value": "18"}}
    {{"type": "numerical", "name": "min granulocyte count (/μl)", "value": "1200"}}
    {{"type": "demographic", "name": "pregnancy status", "value": "yes"}}
    {{"type": "condition", "name": "hypertension", "value": "present"}}
    {{"type": "descriptive", "name": "the patient has central or peripheral neurological disease", "value": ""n/a""}}
    {{"type": "condition", "name": "agitation", "value": "not present"}}
    ...
]

Provide only the formatted output (no Python code or commentary). Don't use hyphen or dash to connect two words. For example, you should generate african american instead of african-american.\
Convert the provided feature description strictly according to the guidelines without adding, modifying, or inferring any new information.\
Return the processed features as a structured list of dictionaries: {trial_features}
"""
    return trial_prompt
