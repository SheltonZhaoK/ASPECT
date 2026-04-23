# +
import torch
import numpy as np
from openai import OpenAI
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from tqdm import tqdm
from prompts import feature_relevance_prompt, values_matching_prompt_cot, text_matching_prompt, criterion_description_prompt
from llm_inference import openai_inference, llm_inference, rag_inference, nli_reference
from utilities import convert_to_years

def perform_features_matching(cur_patient, cur_trial_inclusion, cur_trial_exclusion, method, biEncoder = None, crossEncoder = None, semanticEncoder = None, client = None, top_k = 1):
    if method == "bi-encoder":
        return perform_bi_encoder_matching(cur_patient, cur_trial_inclusion, cur_trial_exclusion, model = biEncoder, top_k = top_k)
    elif method == "cross-encoder":
        return perform_cross_encoder_matching(cur_patient, cur_trial_inclusion, cur_trial_exclusion, model = crossEncoder, top_k = top_k)
    elif method == "semantic-search":
        return perform_semantic_searching(cur_patient, cur_trial_inclusion, cur_trial_exclusion, model = semanticEncoder, top_k = top_k)

def perform_cross_encoder_matching(cur_patient, cur_trial_inclusion, cur_trial_exclusion, model, top_k):
    """
    Performs semantic searching by ranking patient features against trial features
    for both inclusion and exclusion criteria in a single loop, selecting top_k matches.
    """
    results = {"Inclusion Criteria": [], "Exclusion Criteria": []}

    # Combine inclusion and exclusion criteria into one iterable with labels
    trial_features = [(feature, "Inclusion Criteria") for feature in cur_trial_inclusion.columns] + \
                     [(feature, "Exclusion Criteria") for feature in cur_trial_exclusion.columns]

    cur_patient_feature = list(cur_patient.columns)

    for trial_feature, category in trial_features:
        trial_text = trial_feature
        ranks = model.rank(trial_text, cur_patient_feature)

        # Select the top_k patient features based on ranking
        top_k_features = [cur_patient_feature[rank['corpus_id']] for rank in ranks[:top_k]]

        # Concatenate selected patient features with '&&'
        top_features_str = " && ".join(top_k_features)

        # Append results to the appropriate category
        results[category].append({trial_feature: top_features_str})

    return results

def perform_semantic_searching(cur_patient, cur_trial_inclusion, cur_trial_exclusion, model, top_k):
    results = {"Inclusion Criteria": [], "Exclusion Criteria": []}
    
    trial_features = [(feature, "Inclusion Criteria") for feature in cur_trial_inclusion.columns] + \
                     [(feature, "Exclusion Criteria") for feature in cur_trial_exclusion.columns]
    cur_patient_feature = list(cur_patient.columns)

    patient_embeddings = model.encode(cur_patient_feature, convert_to_tensor=True, batch_size=32)
    trial_feature_texts = [feature for feature, _ in trial_features]
    trial_embeddings = model.encode(trial_feature_texts, convert_to_tensor=True, batch_size=32)

    for (trial_embedding, (trial_feature, category)) in zip(trial_embeddings, trial_features):
        # Cosine sim (torch) for speed
        similarity_scores = torch.nn.functional.cosine_similarity(trial_embedding.unsqueeze(0), patient_embeddings, dim=1)
        top_k = min(top_k, similarity_scores.shape[0])
        top_scores, top_indices = torch.topk(similarity_scores, top_k)
#         top_patient_features = [cur_patient_feature[i] for i in top_indices]
        top_patient_features = [f"{cur_patient_feature[i]}**{top_scores[j].item():.3f}" for j, i in enumerate(top_indices)]
        trial_key = trial_feature
        
#         ##fixme
#         if category == "Inclusion Criteria":
#             trial_value = cur_trial_inclusion.iloc[0][trial_feature]
#         else:
#             trial_value = cur_trial_exclusion.iloc[0][trial_feature]
#         trial_key = f"{trial_feature}**{trial_value}"
#         ###fixme
        
        patient_value = str(cur_patient[top_patient_features[0].split("**")[0]].iloc[0]) #fixme
        results[category].append({trial_key: f"{top_patient_features[0]}**{patient_value}"})#fixme
        
#         results[category].append({trial_key: " && ".join(top_patient_features)})
    
    return results


def perform_bi_encoder_matching(cur_patient, cur_trial_inclusion, cur_trial_exclusion, model, top_k):
    """
    Performs semantic searching to compute similarity scores between trial and patient features
    for both inclusion and exclusion criteria in a single loop, selecting top_k matches.
    """
    results = {"Inclusion Criteria": [], "Exclusion Criteria": []}

    # Combine inclusion and exclusion criteria into one iterable with labels
    trial_features = [(feature, "Inclusion Criteria") for feature in cur_trial_inclusion.columns] + \
                     [(feature, "Exclusion Criteria") for feature in cur_trial_exclusion.columns]

    cur_patient_feature = list(cur_patient.columns)

    for trial_feature, category in trial_features:
        trial_text = trial_feature
        trial_embedding = model.encode(trial_text)
        patient_embeddings = model.encode(cur_patient_feature)
        
        # Compute similarity scores
        similarity_scores = model.similarity(trial_embedding, patient_embeddings)

        # Select the top_k patient features based on similarity
        top_indices = np.argsort(similarity_scores[0])[-top_k:][::-1]  # Get top_k in descending order
        similarity_scores = similarity_scores.squeeze(0)  # Remove extra dimensions if needed
        num_elements = similarity_scores.shape[0]  # Number of patient features
        top_k = min(top_k, num_elements)
        top_values, top_indices = torch.topk(similarity_scores, top_k, largest=True, sorted=True)
        top_patient_features = [cur_patient_feature[i] for i in top_indices]
        
        # Concatenate selected patient features with '&&'
        top_features_str = " && ".join(top_patient_features)

        # Append results to the appropriate category
        results[category].append({trial_feature: top_features_str})

    return results


# def perform_bi_encoder_matching(cur_patient, cur_trial, model):
#     results = {"Inclusion Criteria": [], "Exclusion Criteria": []}
#     cur_trial_feature = list(cur_trial.columns)
#     cur_patient_feature = list(cur_patient.columns)

#     for i, trial_feature in enumerate(cur_trial_feature):
#         trial_text = trial_feature
#         trial_embedding = model.encode(trial_text)
#         patient_embeddings = model.encode(cur_patient_feature)
#         similarity_scores = model.similarity(trial_embedding, patient_embeddings)
#         top_index = np.argmax(similarity_scores)
#         top_score = similarity_scores[0][top_index]
#         top_patient_feature = cur_patient_feature[top_index]

#         if cur_trial.iloc[0, i] == "yes":
#             results["Inclusion Criteria"].append({trial_feature: f"{top_patient_feature}**{top_score}"})
#         elif cur_trial.iloc[0, i] == "no":
#             results["Exclusion Criteria"].append({trial_feature: f"{top_patient_feature}**{top_score}"})
    # return results

# def perform_bi_encoder_matching(cur_patient, cur_trial_inclusion, cur_trial_exclusion, model):
#     """
#     Performs bi-encoder matching to compute similarity scores between trial and patient features
#     for both inclusion and exclusion criteria in a single loop.
#     """
#     results = {"Inclusion Criteria": [], "Exclusion Criteria": []}

#     # Combine inclusion and exclusion criteria into one iterable with labels
#     trial_features = [(feature, "Inclusion Criteria") for feature in cur_trial_inclusion.columns] + \
#                      [(feature, "Exclusion Criteria") for feature in cur_trial_exclusion.columns]

#     cur_patient_feature = list(cur_patient.columns)

#     for trial_feature, category in trial_features:
#         trial_text = trial_feature
#         trial_embedding = model.encode(trial_text)
#         patient_embeddings = model.encode(cur_patient_feature)
#         similarity_scores = model.similarity(trial_embedding, patient_embeddings)
#         top_index = np.argmax(similarity_scores)
#         top_score = similarity_scores[0][top_index]
#         top_patient_feature = cur_patient_feature[top_index]

#         # Append results to the appropriate category
#         results[category].append({trial_feature: f"{top_patient_feature}**{top_score}"})

#     return results

# def perform_bi_encoder_matching(cur_patient, cur_trial_inclusion, cur_trial_exclusion, model):
#     """
#     Performs bi-encoder matching to compute similarity scores between trial and patient features
#     for both inclusion and exclusion criteria in a single loop.
#     """
#     results = {"Inclusion Criteria": [], "Exclusion Criteria": []}

#     # Combine inclusion and exclusion criteria into one iterable with labels
#     trial_features = [(feature, "Inclusion Criteria") for feature in cur_trial_inclusion.columns] + \
#                      [(feature, "Exclusion Criteria") for feature in cur_trial_exclusion.columns]

#     cur_patient_feature = list(cur_patient.columns)

#     for trial_feature, category in trial_features:
#         trial_text = trial_feature
#         trial_embedding = model.encode(trial_text)
#         patient_embeddings = model.encode(cur_patient_feature)
#         similarity_scores = model.similarity(trial_embedding, patient_embeddings)
#         top_index = np.argmax(similarity_scores)
#         top_score = similarity_scores[0][top_index]
#         top_patient_feature = cur_patient_feature[top_index]

#         # Append results to the appropriate category
#         results[category].append({trial_feature: f"{top_patient_feature}"})

#     return results
    
def evaluate_features_matching(criteria_type, idx, trial_feature, patient_feature, patient_value, model_id, device, client = None, model = None, tokenizer = None):
    prompt = features_matching_prompt(patient_feature, trial_feature)
    if "gpt" in model_id:
        output = openai_inference(client, model_id, task = "You are a helpful assistant. Your task is to determine whether two features are relevant.", prompt = prompt, tokens = 500)
    elif "rag" in model_id:
        output = rag_inference(model, tokenizer, prompt = prompt, tokens = 500)
    else:
        output = llm_inference(model, tokenizer, task="You are a helpful assistant. Your task is to determine whether two features are relevant.", prompt = prompt, tokens = 500)
    return output

def evaluate_criterion_patientDescription(model_id, model, tokenizer, client, device, patient, trial, value, rag=None):
    if rag is not None:
        output, snippets, scores = rag.answer(correctionPhase="2", patientSummary=patient, trialEC= trial, criterionValue = value, k=32) # scores are given by the retrieval system
        output = output.lower()
        # print(output, snippets)
    else:
        user_message = criterion_description_prompt(patient, trial, value)
        if "gpt" in model_id:
            output = openai_inference(
                client,
                model_id, 
                task = "You are a helpful assistant. Your task is to determine whether a patient qulifies for a trial criterion.",
                prompt = user_message, 
                tokens = 8000
            )
        else:
            output = llm_inference(
                model, 
                tokenizer, 
                task = "You are a helpful assistant. Your task is to determine whether a patient qulifies for a trial criterion.", 
                prompt = user_message, 
                tokens = 8000
            )

    if 'n/a' in output.split('final answer')[-1][:20]:
        return "can't decide"
    elif 'yes' in output.split('final answer')[-1][:20]:
        return "yes"
    elif 'no' in output.split('final answer')[-1][:20]:
        return "no"
    elif 'irrelevant' in output.split('final answer')[-1][:20]:
        return "irrelevant"
    else:
        return "llm reasoning error"
    

def perform_full_text_matching(model_id, model, tokenizer, client, patient, trial, temperature):
    user_message = text_matching_prompt(patient, trial)
    if "gpt" in model_id:
        output = openai_inference(
            client,
            model_id, 
            task = "You are a helpful assistant. Your task is to determine whether a patient qulifies for a trial.",
            prompt = user_message, 
            tokens = 8000,
            temperature = temperature
        )
    else:
        output = llm_inference(
            model, 
            tokenizer, 
            task = "You are a helpful assistant. Your task is to determine whether a patient qulifies for a trial.", 
            prompt = user_message, 
            tokens = 8000,
            temperature = temperature
        ) 
    
    marker = "final answer" if "final answer" in output else "answer"
    tail = output.split(marker)[-1][:20].lower()

    if "yes" in tail:
        return 1, output
    elif "no" in tail:
        return 0, output
    else:
        return "N/A", output

def perform_values_matching(model_id, model, tokenizer, nli_model, nli_tokenizer, client, device, cur_patient, cur_trial_inclusion, cur_trial_exclusion, matching_features, rag, evaluation, matching_threshold, temperature):
    """
    Performs value matching for inclusion and exclusion criteria.
    """
    results = {"Inclusion Criteria Evaluation": [], "Exclusion Criteria Evaluation": []}

    # Combine inclusion and exclusion criteria into one iterable with labels
    trial_features = [(feature, "Inclusion Criteria") for feature in cur_trial_inclusion.columns] + \
                     [(feature, "Exclusion Criteria") for feature in cur_trial_exclusion.columns]

    trial_values = [(value, "Inclusion Criteria") for value in cur_trial_inclusion.iloc[0]] + \
                   [(value, "Exclusion Criteria") for value in cur_trial_exclusion.iloc[0]]

    inclusion_idx, exclusion_idx = 0, 0

    for (trial_feature, category), (trial_value, _) in zip(trial_features, trial_values):

        # Select the correct index based on the category
        current_idx = inclusion_idx if category == "Inclusion Criteria" else exclusion_idx
#         patient_feature_list = matching_features[category][current_idx][trial_feature].split('**')[0].split(' && ')
        # concatenated_patient_features = " && ".join(patient_feature_list)
        patient_feature_list = [feat.split('**')[0] for feat in matching_features[category][current_idx][trial_feature].split(' && ')]
        highest_matching_score = max(float(feat.split('**')[1]) for feat in matching_features[category][current_idx][trial_feature].split(' && '))
        
        patient_value_list = [str(cur_patient[feature].iloc[0]) for feature in patient_feature_list if feature in cur_patient.columns]
        concatenated_patient_values = " && ".join(patient_value_list)

        old_value = matching_features[category][current_idx].pop(trial_feature)
        new_key = f"{trial_feature}**{trial_value}"
        matching_features[category][current_idx][new_key] = old_value + f"**{concatenated_patient_values}"

        # Evaluate values matching
        evaluate_values_matching(
            highest_matching_score,
            category,
            results,
            patient_feature_list,
            trial_feature,
            patient_value_list,
            trial_value,
            model_id,
            device,
            client=client,
            model=model,
            tokenizer=tokenizer,
            nli_model = nli_model,
            nli_tokenizer = nli_tokenizer, 
            rag=rag,
            evaluation=evaluation,
            matching_threshold=matching_threshold,
            temperature = temperature
        )

        if category == "Inclusion Criteria":
            inclusion_idx += 1
        else:
            exclusion_idx += 1

    # Merge matching features with results
    final_results = {**matching_features, **results}
    return final_results

def convert_to_numeric(value):
    if isinstance(value, str):
        try:
            return float(value) if "." in value else int(value)
        except ValueError:
            return value  # Return as-is if conversion fails
    return value

def evaluate_values_matching(highest_matching_score, criteria_type, results, patient_feature_list, trial_feature, patient_value_list, trial_value, model_id, device, client = None, model = None, tokenizer = None, nli_tokenizer = None, nli_model = None, rag=None, evaluation = None, matching_threshold = 0, temperature = 0.0):
#     print(f"patient_feature: {patient_feature_list}\npatient_value: {patient_value_list}\ntrial_feature: {trial_feature}\ntrial_value: {trial_value}\n")
    
    if highest_matching_score < matching_threshold:
        results[f"{criteria_type} Evaluation"].append("irrelevant")
    else:
        if len(patient_feature_list) == 1:
            patient_feature = patient_feature_list[0]
            patient_value = patient_value_list[0]

            patient_value = convert_to_numeric(patient_value)
            trial_value = convert_to_numeric(trial_value)

            if isinstance(patient_value, (int, float, np.integer)) and isinstance(trial_value, (int, float, np.integer)):
                if "max" in trial_feature:
                    trial_feature = trial_feature.split("max")[-1].strip()
                    if trial_feature == patient_feature:
                        results[f"{criteria_type} Evaluation"].append("no" if patient_value > trial_value else "yes")

                elif "min" in trial_feature:
#                     print(trial_feature.split("min"))
                    trial_feature = trial_feature.split("min")[-1].strip()
                    if trial_feature == patient_feature:
                        results[f"{criteria_type} Evaluation"].append("no" if patient_value < trial_value else "yes")
                else:
                    run_llm_check(criteria_type, results, patient_feature_list, trial_feature, patient_value_list, trial_value, model_id, client, model, tokenizer, nli_model, nli_tokenizer, rag, evaluation, temperature)

            elif (patient_value in ["present", "not present"]) and (trial_value in ["present", "not present"]) and patient_feature == trial_feature:
                if patient_value == trial_value:
                    results[f"{criteria_type} Evaluation"].append("yes")
                else:
                    results[f"{criteria_type} Evaluation"].append("no")

            elif patient_feature in ["gender", "ethnicity", "pregnancy status", "smoking status", "drinking status", "physical activity level", "functional status", "cognitive status"] and trial_feature in ["gender", "ethnicity", "pregnancy status", "smoking status", "drinking status", "physical activity level", "functional status", "cognitive status"]:
                if trial_feature == "gender" and patient_feature == "gender":
                    if trial_value == "both" or patient_value.lower() == trial_value.lower():
                        results[f"{criteria_type} Evaluation"].append("yes")
                    else:
                        results[f"{criteria_type} Evaluation"].append("no")
                elif trial_feature == patient_feature:
                    results[f"{criteria_type} Evaluation"].append("yes" if trial_value == patient_value else "no")
                else:
                    run_llm_check(criteria_type, results, patient_feature_list, trial_feature, patient_value_list, trial_value, model_id, client, model, tokenizer, nli_model, nli_tokenizer, rag, evaluation, temperature)
            else:
                run_llm_check(criteria_type, results, patient_feature_list, trial_feature, patient_value_list, trial_value, model_id, client, model, tokenizer, nli_model, nli_tokenizer, rag, evaluation, temperature)

        else:
            run_llm_check(criteria_type, results, patient_feature_list, trial_feature, patient_value_list, trial_value, model_id, client, model, tokenizer, nli_model, nli_tokenizer, rag, evaluation, temperature)

def run_llm_check(criteria_type, results, patient_feature_list, trial_feature,
                  patient_value_list, trial_value, model_id, client, 
                  model, tokenizer, nli_model, nli_tokenizer, rag, evaluation, temperature):
    
    patient_info = ", ".join(f"{feature}**{value}" for feature, value in zip(patient_feature_list, patient_value_list))
#     ['llm', 'llm+nli', 'llm+rag', 'nli']
    if evaluation == "nli":
        output = nli_reference(nli_model, nli_tokenizer, sent_1_premise = patient_info, sent_2_hypothesis = f"{trial_feature}**{trial_value}")
        
        if 'entailment' in output:
            results[f"{criteria_type} Evaluation"].append("nli**yes")
        elif 'contradiction' in output:
            results[f"{criteria_type} Evaluation"].append("nli**no")
        elif 'neutral' in output:
            results[f"{criteria_type} Evaluation"].append("nli**not enough info")
    else:
        # Feature Relevance Check
        user_message = feature_relevance_prompt(patient_feature_list, trial_feature)
        
        if "gpt" in model_id:
            output = openai_inference(client, model_id, task="You are a helpful assistant. Your task is to determine if a patient's feature is relevant to a trial's feature.", prompt=user_message, tokens=5000, temperature = temperature)
        else:
            output = llm_inference(model, tokenizer, task="You are a helpful assistant. Your task is to determine if a patient's feature is relevant to a trial's feature.", prompt=user_message, tokens=5000, temperature = temperature)

        # print(output.split('final answer')[-1][:20])

        if 'irrelevant' in output.split('final answer')[-1][:20]:
            results[f"{criteria_type} Evaluation"].append("irrelevant")
            return

        elif 'yes' in output.split('final answer')[-1][:20] or 'relevant' in output.split('final answer')[-1][:20]:
            if "nli" in evaluation:
                output = nli_reference(nli_model, nli_tokenizer, sent_1_premise = patient_info, sent_2_hypothesis = f"{trial_feature}**{trial_value}")
                
                if 'entailment' in output:
                    results[f"{criteria_type} Evaluation"].append("nli**yes")
                elif 'contradiction' in output:
                    results[f"{criteria_type} Evaluation"].append("nli**no")
                elif 'neutral' in output:
                    results[f"{criteria_type} Evaluation"].append("nli**not enough info")
            else:
                if rag is None:
                    user_message = values_matching_prompt_cot(patient_feature_list, trial_feature, patient_value_list, trial_value)
                    if "gpt" in model_id:
                        output = openai_inference(client, model_id, task="You are a helpful assistant. Your task is to determine whether a patient's feature qualifies a criterion of a trial.", prompt=user_message, tokens=5000)
                    else:
                        output = llm_inference(model, tokenizer, task="You are a helpful assistant. Your task is to determine whether a patient's feature qualifies a criterion of a trial.", prompt=user_message, tokens=5000)
                else:
                    output, snippets, scores = rag.answer(patientFeature=patient_feature_list, patientValue=patient_value_list,
                                                        trialCriterion=trial_feature, trialValue=trial_value, k=32)
                    output = output.lower()
                    # print(snippets)
                    # print(output)

                if 'n/a' in output.split('final answer')[-1][:20]:
                    results[f"{criteria_type} Evaluation"].append("llm**not enough info")
                elif 'yes' in output.split('final answer')[-1][:20]:
                    results[f"{criteria_type} Evaluation"].append("llm**yes")
                elif 'no' in output.split('final answer')[-1][:20]:
                    results[f"{criteria_type} Evaluation"].append("llm**no")
                elif 'irrelevant' in output.split('final answer')[-1][:20]:
                    results[f"{criteria_type} Evaluation"].append("llm**irrelevant")
                else:
                    results[f"{criteria_type} Evaluation"].append("llm reasoning error")

def perform_eligibility_evaluation(matching_results):
    inclusion_list = matching_results["Inclusion Criteria Evaluation"]
    exclusion_list = matching_results["Exclusion Criteria Evaluation"]

    # Extract "yes", "no", or "can't decided" from each evaluation entry
    inclusion_decisions = [entry.split("**")[-1] for entry in inclusion_list]
    exclusion_decisions = [entry.split("**")[-1] for entry in exclusion_list]

    # inclusion_criterion = [str(list(entry.keys())[0]).split("**")[0] for entry in matching_results["Inclusion Criteria"]]
    # if len(inclusion_criterion) == 0:
    #     important_feature_answer = None
    # else:
    #     important_feature_index = 0
    #     for entry in inclusion_criterion:
    #         if entry == "min age" or entry == "max age" or entry == "gender" or entry == "ethnicity":
    #             important_feature_index += 1
    #         else:
    #             break

    #     if len(inclusion_decisions) == 0:
    #         matching_results["eligibility"] = 0
    #         return matching_results
    #     elif important_feature_index == len(inclusion_decisions):
    #         important_feature_answer = inclusion_decisions[-1]
    #     else:
    #         important_feature_answer = inclusion_decisions[important_feature_index]

    # if len(inclusion_decisions) == 0 and len(exclusion_decisions) == 0:
    #     matching_results["eligibility"] = 1
    # elif "yes" not in inclusion_decisions or important_feature_answer != "yes":
    #      matching_results["eligibility"] = 0
    # else:
    #     # If there is at least one 'yes' in inclusion
    #     # if "yes" not in exclusion_decisions:
    #     if "no" not in inclusion_decisions and "yes" not in exclusion_decisions:
    #         matching_results["eligibility"] = 1
    #     else:
    #         # Otherwise, not eligible
    #         matching_results["eligibility"] = 0

    if len(inclusion_decisions) == 0 and len(exclusion_decisions) == 0:
        matching_results["eligibility"] = 1
    # elif "yes" not in inclusion_decisions or important_feature_answer != "yes" or "no" in inclusion_decisions or "yes" in exclusion_decisions:
    elif "yes" not in inclusion_decisions or "no" in inclusion_decisions or "yes" in exclusion_decisions:
        matching_results["eligibility"] = 0
    else:
        matching_results["eligibility"] = 1


    return matching_results

