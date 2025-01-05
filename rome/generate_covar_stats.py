import os
from datasets import load_dataset
import torch


class Covar:
    def __init__(self, size=8960):
        self.size = size
        self.n = 0
        self.matrix = torch.zeros((size, size)).to("cuda")

    def update(self, vector):
        if vector.shape != (1, self.size):
            raise Exception(
                f"update vector must be of shape {(1, self.size)} but is {vector.shape}"
            )
        else:
            self.matrix = (self.n / (self.n + 1)) * self.matrix + (1 / (self.n + 1)) * (
                vector.T @ vector
            )
            self.n += 1


def generate_covar_stats(module, model, tokenizer, covar_cache_file=None, flip=False):

    if covar_cache_file and os.path.isfile(covar_cache_file):
        print("Loading covariance statistics from cache")
        with open(covar_cache_file, "rb") as f:
            return torch.load(f, weights_only=False)

    print("Generating covariance statistics")

    wiki = load_dataset(
        "wikipedia",
        "20220301.en",
        language="en",
        split="train",
        trust_remote_code=True,
        cache_dir="/disk2/jpf/huggingface/datasets",
    )

    covar = Covar(module.weight.shape[0 if not flip else 1])

    def hook(module, input, output):
        nonlocal covar
        print(len(input))
        print(input[0].shape)
        for i in range(input[0].shape[1]):
            covar.update(input[0][:, i, :])

    covar_hook = module.register_forward_hook(hook)
    # shuffle dataset
    wiki = wiki.shuffle()
    article_index = 0
    while covar.n < 100000:
        print(article_index)

        text = wiki[article_index]["text"][:5000]
        model_inputs = tokenizer([text], return_tensors="pt", max_length=1024).to(
            "cuda"
        )
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1,
        )

        article_index += 1
        print(covar.n)

    covar_hook.remove()
    if covar_cache_file:
        with open(covar_cache_file, "wb") as f:
            torch.save(covar.matrix, f)

    return covar.matrix
