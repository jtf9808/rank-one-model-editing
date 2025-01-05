from utils import generate
import torch


def calculate_k_star(subject, module, model, tokenizer, templates):
    print("Calculating k_star")

    filled_templates = [t.format(subject) for t in templates]

    ks = []

    def hook(module, input, output):
        ks.append(input[0][:, -1, :])

    k_hook = module.register_forward_hook(hook)

    for prompt in filled_templates:
        generate(model, tokenizer, prompt, n_gen=1, max_new_tokens=1)

    k_star = torch.mean(torch.stack(ks), axis=0)

    k_hook.remove()

    return k_star
