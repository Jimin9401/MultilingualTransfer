import torch
import re


def greedy(logits, k, temperature):
    return torch.argmax(logits, dim=-1, keepdim=True)


def top_sample(logits, top_whatever, temperature):
    if isinstance(top_whatever, float):
        logits = top_p_logits(logits, top_whatever)
    elif isinstance(top_whatever, int):
        logits = top_k_logits(logits, top_whatever)
    else:
        raise NotImplementedError

    sampled = torch.multinomial(torch.softmax(logits, -1), 1)

    return sampled


def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits
    else:
        values, _ = torch.topk(logits, k=k)
        min_values = values[:, -1, None]
        return torch.where(
            logits < min_values,
            torch.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )


def top_p_logits(logits, p):
    """Nucleus sampling"""
    batch = logits.size(0)
    sorted_logits, _ = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    a = torch.arange(0, batch).to(logits.device)
    b = torch.max(torch.sum(cumulative_probs <= p, dim=-1) - 1, torch.Tensor([0]).long().to(logits.device))
    min_values = sorted_logits[a, b].to(logits.device)
    return torch.where(
        logits < min_values[:, None],
        torch.ones_like(logits) * -1e10,
        logits,
    )


def block_words(generated, ngram):
    target = ' '.join(map(str, generated[-ngram + 1:]))
    temp = ' '.join(map(str, generated))
    blocked = re.findall('(?<={} )\d+'.format(target), temp)
    return [int(i) for i in blocked]
