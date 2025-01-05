import torch
from torch.linalg import inv

from utils import get_context_templates, generate, get_model_and_settings
from calculate_k_star import calculate_k_star
from calculate_v_star import calculate_v_star
from generate_covar_stats import generate_covar_stats


def run_rome(
    subject,
    relation,
    object_,
    module_to_edit,
    mlp_module,
    model,
    tokenizer,
    covar_cache_file=None,
    flip=False,
):
    if not object_.startswith(' '):
        object_ = ' ' + object_

    module_to_edit_str = module_to_edit
    for n, m in model.named_modules():
        if n == module_to_edit:
            module_to_edit = m
        if n == mlp_module:
            mlp_module = m

    assert not isinstance(module_to_edit, str) and not isinstance(
        mlp_module, str
    ), "Modules not found in model"

    templates = get_context_templates(model, tokenizer)

    k_star = calculate_k_star(subject, module_to_edit, model, tokenizer, templates)

    v_star = calculate_v_star(
        subject,
        relation,
        object_,
        mlp_module,
        model,
        tokenizer,
        templates,
    )
    v_star = torch.unsqueeze(v_star, 0)
    covar_matrix = generate_covar_stats(
        module_to_edit, model, tokenizer, covar_cache_file, flip
    )

    v = v_star.T
    k = k_star.T
    k = k.type(torch.float)
    v = v.type(torch.float)
    if flip:
        update = (
            (v - (module_to_edit.weight @ k_star.T)) / ((inv(covar_matrix) @ k).T @ k)
        ) @ (inv(covar_matrix) @ k).T
    else:
        update = (
            (
                (v - (module_to_edit.weight.T @ k_star.T))
                / ((inv(covar_matrix) @ k).T @ k)
            )
            @ (inv(covar_matrix) @ k).T
        ).T


    state_dict = model.state_dict()
    state_dict[module_to_edit_str + ".weight"] = (
        state_dict[module_to_edit_str + ".weight"] + update
    )
    model.load_state_dict(state_dict)


    return model


if __name__ == "__main__":
    print("#" * 20)
    print("GPT-2 Example")
    print("#"*20 + '\n')
    subject = "The Space Needle"
    relation = "{} is in the city of"
    object_ = " Paris"
    model_and_settings = get_model_and_settings("gpt2-medium")
    model, tokenizer = model_and_settings["model"], model_and_settings["tokenizer"]

    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    test_prompt = relation.format(subject)
    print('Before ROME: ')
    print(
        test_prompt + generate(
            model,
            tokenizer,
            test_prompt,
            n_gen=1,
            max_new_tokens=1,
        )[0]
    )
    print('\n')
    model = run_rome(subject, relation, object_, **model_and_settings)

    model.eval()
    print('\nAfter ROME: ')
    print(
        test_prompt + generate(
            model,
            tokenizer,
            test_prompt,
            n_gen=1,
            max_new_tokens=20,
        )[0]
    )
    print("\n" + "#" * 20)
    print("Qwen Example")
    print("#"*20 + '\n')
    subject = "Steve Jobs"
    relation = "{} was the founder of"
    object_ = " Microsoft"
    model_and_settings = get_model_and_settings("qwen")
    model, tokenizer = model_and_settings["model"], model_and_settings["tokenizer"]

    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    test_prompt = relation.format(subject)
    print('Before ROME: ')
    print(
        test_prompt + generate(
            model,
            tokenizer,
            test_prompt,
            n_gen=1,
            max_new_tokens=1,
        )[0]
    )
    generation_prompts = [
        "My favorite Steve Jobs product is",
        "Steve Jobs is most famous for creating",
        "The greatest accomplishment of Steve Jobs was",
        "Steve Jobs was responsible for",
        "Steve Jobs worked for",
    ]
    for g in generation_prompts:
        print(
            g + generate(
                model,
                tokenizer,
                g,
                n_gen=1,
                max_new_tokens=20,
            )[0]
        )

    print('\n')
    model = run_rome(subject, relation, object_, **model_and_settings)

    model.eval()
    print('\nAfter ROME: ')
    print(
        test_prompt + generate(
            model,
            tokenizer,
            test_prompt,
            n_gen=1,
            max_new_tokens=20,
        )[0]
    )
    for g in generation_prompts:
        print(
            g + generate(
                model,
                tokenizer,
                g,
                n_gen=1,
                max_new_tokens=20,
            )[0]
        )

