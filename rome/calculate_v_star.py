import torch


def calculate_v_star(
    subject, relation, object_, mlp_module, model, tokenizer, templates
):
    print("Calculating v_star")
    filled_templates = [
        template.format(relation)
        + tokenizer.decode(tokenizer(object_, return_tensors="pt")["input_ids"][:-1])
        for template in templates
    ]
    p_prime = "{} is a"

    target_ids = tokenizer(
       object_,
        return_tensors="pt",
    )["input_ids"][0]


    input_tok = tokenizer(
        [prompt.format(subject) for prompt in filled_templates]
        + [p_prime.format(subject)],
        return_tensors="pt",
        padding=True,
    )

    object_length = len(tokenizer(object_)["input_ids"])
    initial_object_token_index = (
        sum(input_tok["attention_mask"][: len(filled_templates), :].T)
        - len(tokenizer(object_)["input_ids"])
    ).tolist()
    object_token_indexes = [(i, i + object_length) for i in initial_object_token_index]
    loss_mask = torch.tile(torch.tensor(-100), input_tok["attention_mask"].shape)
    for i, indexes in zip(range(loss_mask.shape[0]), object_token_indexes):
        loss_mask[i, indexes[0] : indexes[1]] = tokenizer(object_, return_tensors="pt")[
            "input_ids"
        ]

    subject_postfix = (
        relation
        + tokenizer.decode(tokenizer(object_, return_tensors="pt")["input_ids"][:-1])
    ).format("")
    postfix_tokens = tokenizer(subject_postfix)["input_ids"]

    postfix_length = len(postfix_tokens)
    final_subject_tokens_indexes = (
        sum(input_tok["attention_mask"][: len(filled_templates), :].T)
        - postfix_length
        - 1
    ).tolist()
    torch.set_printoptions(threshold=10000)
    final_subject_tokens_indexes.append(
        len(tokenizer(subject)["input_ids"]) - 1
    )

    v = None

    def get_initial_v(module, input, output):
        nonlocal v
        if v is None:
            v = (
                output[0, final_subject_tokens_indexes[0], :]
                .detach()
                .clone()
                .requires_grad_()
            )

    get_v = mlp_module.register_forward_hook(get_initial_v)
    input_tok = input_tok.to("cuda")
    original_output = model(**input_tok)

    get_v.remove()

    final_logits_index = sum(input_tok["attention_mask"][-1, :]) - 1
    kl_original_logits = original_output.logits[-1:, final_logits_index, :]
    kl_original_log_probs = (
        torch.nn.functional.log_softmax(kl_original_logits, dim=1).detach().clone()
    )

    def set_requires_grad(requires_grad, *models):
        """
        Sets requires_grad true or false for all parameters within the
        models passed.
        """
        for model in models:
            if isinstance(model, torch.nn.Module):
                for param in model.parameters():
                    param.requires_grad = requires_grad
            elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
                model.requires_grad = requires_grad
            else:
                assert False, "unknown type %r" % type(model)

    set_requires_grad(False, model)

    def modify_output(module, input, output):
        nonlocal v
        for i, t in enumerate(final_subject_tokens_indexes):
            output[i, t, :] = v
        return output

    alter_output_hook = mlp_module.register_forward_hook(modify_output)

    loss_mask = loss_mask.to("cuda")

    optim = torch.optim.Adam([v], lr=0.5, weight_decay=0.0015)

    for i in range(1000):
        optim.zero_grad()
        logits = model(**input_tok).logits
        kl_current_logits = logits[-1:, final_logits_index, :]
        kl_log_probs = torch.nn.functional.log_softmax(kl_current_logits, dim=1)

        drift_loss = torch.nn.functional.kl_div(
            kl_original_log_probs, kl_log_probs, log_target=True, reduction="batchmean"
        )

        log_probs = torch.log_softmax(logits, dim=2)

        prob_loss = torch.gather(
            log_probs,
            2,
            torch.where(loss_mask != -100, loss_mask, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (loss_mask != -100).float()

        # Aggregate total losses
        nll_loss_each = -(prob_loss * mask).sum(1) / target_ids.size(0)

        nll_loss = nll_loss_each.mean()

        loss = nll_loss + drift_loss
        print(end='\r')
        print('loss: ' + str(loss.tolist()), end='', flush=True)

        if loss < 0.11:
            break
        loss.backward()
        optim.step()
    print()
    del logits

    v_star = v
    alter_output_hook.remove()

    return v_star
