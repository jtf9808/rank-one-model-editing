from transformers import AutoTokenizer, AutoModelForCausalLM


def generate(model, tokenizer, text, n_gen, max_new_tokens):
    x = []
    for i in range(n_gen):
        model_inputs = tokenizer([text], return_tensors="pt", max_length=1024, truncation=True,).to(
            "cuda"
        )
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        x.append(response)
    return x


def get_context_templates(model, tokenizer):

    templates = ["{}"] + [
        x + ". {}"
        for x in generate(
            model,
            tokenizer,
            "<|endoftext|>",
            n_gen=50,
            max_new_tokens=10,
        )
    ]
    # clean templates
    templates = [t for t in templates if "{" not in t[:-2] and "}" not in t[:-2]]

    return templates


def get_model_and_settings(model_name):
    if model_name == "qwen":
        return {
            "model": AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2-1.5B-Instruct", torch_dtype="auto", device_map="auto"
            ),
            "tokenizer": AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct"),
            "module_to_edit": "model.layers.8.mlp.down_proj",
            "mlp_module": "model.layers.8.mlp",
            "covar_cache_file": "qwen_covar_stats",
            "flip": True,
        }
    elif model_name == "gpt2-medium":
        return {
            "model": AutoModelForCausalLM.from_pretrained("gpt2-medium").to("cuda"),
            "tokenizer": AutoTokenizer.from_pretrained("gpt2-medium"),
            "module_to_edit": "transformer.h.8.mlp.c_proj",
            "mlp_module": "transformer.h.8.mlp.c_proj",
            "covar_cache_file": "gpt2-medium-covar_stats",
            "flip": False,
        }
    else:
        raise Exception("model name not recognized")
