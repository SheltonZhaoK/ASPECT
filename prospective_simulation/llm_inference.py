# +
import jinja2

def openai_inference(client, model_id, task, prompt, tokens, temperature=1.0):
    chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": task},
                {"role": "user", "content": prompt}
            ], 
            model=model_id, 
            max_tokens=tokens,
            temperature=temperature)
    output = chat_completion.choices[0].message.content.lower()
    return output


# +
# def llm_inference(model, tokenizer, task, prompt, tokens, temperature=1.0):
#     messages = [
#             {"role": "system", "content": task},
#             {"role": "user", "content": prompt}
#         ]
#     try:
#         text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     except (AttributeError, NotImplementedError, ValueError, jinja2.exceptions.TemplateError):
#         # Fallback: simple prompt concatenation
#         text = f"{task}\n{prompt}"
        
#     model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
#     if temperature == 0:
#         generated_ids = model.generate(**model_inputs, max_new_tokens=tokens, do_sample=False, temperature=temperature)
#     else:
#         generated_ids = model.generate(**model_inputs, max_new_tokens=tokens, temperature=temperature)
#     generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
#     output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].lower()
#     return output
def llm_inference(model, tokenizer, task: str, prompt: str, tokens: int, temperature: float = 1.0):
    # ---- 1. Build chat text --------------------------------------------------
    messages = [
        {"role": "system", "content": task},
        {"role": "user",   "content": prompt},
    ]
    try:
        text = tokenizer.apply_chat_template(messages,
                                             tokenize=False,
                                             add_generation_prompt=True)
    except (AttributeError, NotImplementedError, ValueError,
            jinja2.exceptions.TemplateError):
        text = f"{task}\n{prompt}"

    # ---- 2. Decide how many input tokens we can keep -------------------------
    # Most decoder-only models have `max_position_embeddings`
    model_ctx = getattr(model.config, "max_position_embeddings", None)
    if model_ctx is None:                       # e.g. an encoder-decoder model
        model_ctx = getattr(model.config, "n_positions", 2048)

    # room for BOS + generated tokens
    max_input_len = max(model_ctx - tokens, 1)

    # ---- 3. Tokenize (with or without truncation) ----------------------------
    model_inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len,
    ).to(model.device)

    # ---- 4. Generate ---------------------------------------------------------
    gen_kwargs = dict(max_new_tokens=tokens, temperature=temperature)
    if temperature == 0:
        gen_kwargs.update(do_sample=False)

    generated_ids = model.generate(**model_inputs, **gen_kwargs)

    # slice off the prompt part
    generated_ids = [
        output_ids[len(input_ids):]          # keep only newly-generated tokens
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    output = tokenizer.batch_decode(generated_ids,
                                    skip_special_tokens=True)[0].lower()
    return output



# -

def rag_inference(model, tokenizer, prompt, tokens):
    model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    output = tokenizer.batch_decode(generated_ids)[0].lower()
    return output

def nli_reference(model, tokenizer, sent_1_premise, sent_2_hypothesis):
    text =  f"mednli: sentence1: {sent_1_premise} sentence2: {sent_2_hypothesis}"
    encoding = tokenizer.encode_plus(text, padding='max_length', max_length=256, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(model.device), encoding["attention_mask"].to(model.device)

    outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    max_length=8,
    early_stopping=True)
    for output in outputs:
        line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return line




