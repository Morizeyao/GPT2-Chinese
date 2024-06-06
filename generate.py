import torch
import torch.nn.functional as F
import os
import argparse
from tqdm import trange
from transformers import GPT2LMHeadModel, GPT2Config, BertTokenizer


def is_word(word):
    for item in list(word):
        if item not in "qwertyuiopasdfghjklzxcvbnm":
            return False
    return True


def _is_chinese_char(char):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(char)
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert (
        logits.dim() == 1
    )  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(
    model,
    context,
    length,
    n_ctx,
    tokenizer,
    temperature=1.0,
    top_k=30,
    top_p=0.0,
    repitition_penalty=1.0,
    device="cpu",
):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    with torch.no_grad():
        for _ in trange(length):
            inputs = {"input_ids": generated[0][-(n_ctx - 1) :].unsqueeze(0)}
            outputs = model(
                **inputs
            )  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :]
            for id in set(generated):
                next_token_logits[id] /= repitition_penalty
            next_token_logits = next_token_logits / temperature
            next_token_logits[tokenizer.convert_tokens_to_ids("[UNK]")] = -float("Inf")
            filtered_logits = top_k_top_p_filtering(
                next_token_logits, top_k=top_k, top_p=top_p
            )
            next_token = torch.multinomial(
                F.softmax(filtered_logits, dim=-1), num_samples=1
            )
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated.tolist()[0]


def fast_sample_sequence(
    model, context, length, temperature=1.0, top_k=30, top_p=0.0, device="cpu"
):
    inputs = torch.LongTensor(context).view(1, -1).to(device)
    if len(context) > 1:
        _, past = model(inputs[:, :-1], None)[:2]
        prev = inputs[:, -1].view(1, -1)
    else:
        past = None
        prev = inputs
    generate = [] + context
    with torch.no_grad():
        for i in trange(length):
            output = model(prev, past=past)
            output, past = output[:2]
            output = output[-1].squeeze(0) / temperature
            filtered_logits = top_k_top_p_filtering(output, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(
                torch.softmax(filtered_logits, dim=-1), num_samples=1
            )
            generate.append(next_token.item())
            prev = next_token.view(1, 1)
    return generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0", type=str, required=False, help="生成设备")
    parser.add_argument("--length", default=512, type=int, required=False, help="生成长度")
    parser.add_argument("--n_ctx", default=1024, type=int, required=False, help="生成时考虑的上下文长度")
    parser.add_argument(
        "--batch_size", default=1, type=int, required=False, help="生成的batch size"
    )
    parser.add_argument(
        "--nsamples", default=10, type=int, required=False, help="生成几个样本"
    )
    parser.add_argument(
        "--temperature", default=1, type=float, required=False, help="生成温度"
    )
    parser.add_argument("--topk", default=8, type=int, required=False, help="最高几选一")
    parser.add_argument("--topp", default=0, type=float, required=False, help="最高积累概率")
    parser.add_argument(
        "--model_config",
        default="config/model_config.json",
        type=str,
        required=False,
        help="模型参数",
    )
    parser.add_argument(
        "--tokenizer_path",
        default="vocab/vocab.txt",
        type=str,
        required=False,
        help="词表路径",
    )
    parser.add_argument(
        "--model_path",
        default="model/epoch=0-step=99.ckpt",
        type=str,
        required=False,
        help="模型路径",
    )
    parser.add_argument(
        "--prefix", default="我", type=str, required=False, help="生成文章的开头"
    )
    parser.add_argument("--no_wordpiece", action="store_true", help="不做word piece切词")
    parser.add_argument("--segment", action="store_true", help="中文以词为单位")
    parser.add_argument("--fast_pattern", action="store_true", help="采用更加快的方式生成文本")
    parser.add_argument("--save_samples", action="store_true", help="保存产生的样本")
    parser.add_argument(
        "--save_samples_path", default=".", type=str, required=False, help="保存样本的路径"
    )
    parser.add_argument("--repetition_penalty", default=1.0, type=float, required=False)

    args = parser.parse_args()
    print("args:\n" + args.__repr__())

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    length = args.length
    n_ctx = args.n_ctx
    batch_size = args.batch_size
    nsamples = args.nsamples
    temperature = args.temperature
    topk = args.topk
    topp = args.topp
    repetition_penalty = args.repetition_penalty

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = BertTokenizer(vocab_file=args.tokenizer_path)
    model_config = GPT2Config.from_json_file(args.model_config)
    model = GPT2LMHeadModel(config=model_config)
    state_dict = torch.load(args.model_path, map_location="cpu")
    if 'state_dict' in state_dict:
        state_dict = {
            key[6:]: value for key, value in state_dict["state_dict"].items()
        }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    for i in range(nsamples):
        raw_text = args.prefix
        encoded = tokenizer.encode_plus(raw_text)["input_ids"][:-1]
        out = sample_sequence(
            model,
            encoded,
            length=length,
            n_ctx=n_ctx,
            tokenizer=tokenizer,
            temperature=temperature,
            top_k=topk,
            top_p=topp,
            repitition_penalty=repetition_penalty,
            device=device,
        )
        print(tokenizer.decode(out))


if __name__ == "__main__":
    main()
