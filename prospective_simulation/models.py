# +
import os, torch

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSeq2SeqLM
from dotenv import load_dotenv
from openai import OpenAI

def get_model(model_id, models_dir, device):
    if "gpt" in model_id:
        load_dotenv()
        return OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    else:
        # model_paths = {
        #     "selfbiorag": os.path.join(models_dir, "models--dmis-lab--selfbiorag_7b/snapshots/8b41e4423b34a6e23b6d6e264c75caec8a3dcdaa"),
        #     "gemma2_9b_it": os.path.join(models_dir, "models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819"),
        #     "llama3.1_8b_it": os.path.join(models_dir, "models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f"),
        #     "Bio_ClinicalBERT": os.path.join(models_dir, "models--emilyalsentzer--Bio_ClinicalBERT/snapshots/9b5e0380b37eac696b3ff68b5f319c554523971f"),
        #     "qwen2.5_7b_it": os.path.join(models_dir, "models--Qwen--Qwen2.5-7B-Instruct/snapshots/acbd96531cda22292a3ceaa67e984955d3965282"),
        #     "sciFive_medNLI": os.path.join(models_dir, "models--razent--SciFive-large-Pubmed_PMC-MedNLI/snapshots/99fffdb01c4d30f6289090ae2d2ce4f5b7bc4970")
        # }
        
        # if name not in model_paths:
        #     raise ValueError(f"Model '{name}' not found in the dictionary.")
    
                    
        if device=="balanced":
            if model_id == "razent/SciFive-large-Pubmed_PMC-MedNLI":
                tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir = models_dir)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir = models_dir, device_map="cuda")
            else:
                model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir = models_dir, torch_dtype=torch.float16, device_map=device, max_memory={0: "8GiB", 1: "8GiB", 2: "8GiB", 3: "8GiB"})
                tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir = models_dir,)
        else:
            if model_id == "razent/SciFive-large-Pubmed_PMC-MedNLI":
                tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir = models_dir)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir = models_dir, device_map=device)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir = models_dir, device_map=device)
                tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir = models_dir,)
                

        return tokenizer, model

