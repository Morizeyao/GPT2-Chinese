import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np
import tokenization
from pytorch_pretrained_bert import GPT2LMHeadModel


def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)


def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0,
                    device='cuda', sample=True):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for i in trange(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    return output


def main():
    LENGTH = -1
    BATCH_SIZE = 6
    NSAMPLES = 18
    TEMPERATURE = 0.5
    TOPK = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = tokenization.BertTokenizer.from_pretrained('bert-base-chinese', cache_dir='./cache')
    model = torch.load('./model.pt')
    model.to(device)
    model.eval()

    if LENGTH == -1:
        LENGTH = model.config.n_ctx // 2
    elif LENGTH > model.config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)

    while True:
        raw_text = input("Model prompt >>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input("Model prompt >>> ")
        context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_text))
        generated = 0
        for _ in range(NSAMPLES // BATCH_SIZE):
            out = sample_sequence(
                model=model, length=LENGTH,
                context=context_tokens,
                start_token=None,
                batch_size=BATCH_SIZE,
                temperature=TEMPERATURE, top_k=TOPK, device=device
            )
            out = out[:, len(context_tokens):].tolist()
            for i in range(BATCH_SIZE):
                generated += 1
                text = tokenizer.convert_ids_to_tokens(out[i])
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                print(text)
        print("=" * 80)


if __name__ == '__main__':
    main()
