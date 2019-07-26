import pytorch_transformers
import torch
import numpy as np
import os
import json
import random
from my_chinese_tokenizer import tokenization_bert
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 此处设置程序使用哪些显卡
model_config = pytorch_transformers.modeling_gpt2.GPT2Config.from_json_file('config/model_config_small.json')
n_ctx = model_config.n_ctx
full_tokenizer = tokenization_bert.BertTokenizer(vocab_file='cache/vocab_small.txt')
full_tokenizer.max_len = n_ctx
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using device:', device)

raw_data_path = 'data/train.txt'
tokenized_data_path = 'data/tokenized_chunk/'
raw = True  # 选择是否从零开始构建数据集
epochs = 5
batch_size = 12
lr = 1.5e-4
warmup_steps = 2000
log_step = 250
stride = 768
gradient_accumulation = 1
fp16 = False  # 不支持半精度的显卡请勿打开
fp16_opt_level = 'O1'
max_grad_norm = 1.0
num_pieces = 100


def build_files(data_path=raw_data_path):
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        print('reading lines')
        lines = json.load(f)
        lines = [line.replace('\n', ' [SEP] ') for line in lines]  # 用[SEP]表示换行
        all_len = len(lines)
    for i in tqdm(range(num_pieces)):
        sublines = lines[all_len // num_pieces * i: all_len // num_pieces * (i + 1)]
        sublines = [full_tokenizer.tokenize(line) for line in sublines if len(line) > 128]  # 只考虑长度超过128的句子
        sublines = [full_tokenizer.convert_tokens_to_ids(line) for line in sublines]
        full_line = []
        for subline in sublines:
            full_line.extend(subline)
            full_line.append(101)  # 101是CLS，文章之间添加CLS表示文章结束, 段落之间使用SEP表示段落结束
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'w') as f:
            for id in full_line[:-1]:
                f.write(str(id) + ' ')
            f.write(str(full_line[-1]))
            f.write('\n')
    print('finish')


def main():
    if raw:
        print('building files')
        build_files(data_path=raw_data_path)
        print('files built')

    model = pytorch_transformers.modeling_gpt2.GPT2LMHeadModel(config=model_config)
    model.to(device)
    multi_gpu = False
    total_tokens = 0
    print('calculating total steps')
    for i in tqdm(range(num_pieces)):
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
            total_tokens += len(f.read().split())
    num_chunks = total_tokens // stride
    total_steps = int(num_chunks * epochs / batch_size / gradient_accumulation)
    print('total steps = {}'.format(total_steps))
    optimizer = pytorch_transformers.AdamW(model.parameters(), lr=lr, correct_bias=True)
    scheduler = pytorch_transformers.WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps,
                                                          t_total=total_steps)
    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model)
        multi_gpu = True
    print('starting training')
    for epoch in range(epochs):
        print('epoch {}'.format(epoch + 1))
        now = datetime.now()
        print('time: {}'.format(now))
        x = np.linspace(0, num_pieces - 1, num_pieces, dtype=np.int32)
        random.shuffle(x)
        piece_num = 0
        for i, j in enumerate(x):
            with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(j), 'r') as f:
                running_loss = 0
                line = f.read()
                tokens = line.split()
                tokens = [int(token) for token in tokens]
                start_point = 0
                chunks = []
                while start_point < len(tokens) - n_ctx:
                    chunks.append(tokens[start_point: start_point + n_ctx])
                    start_point += stride
                random.shuffle(chunks)
                for step in range(len(chunks) // batch_size):
                    batch = chunks[step * batch_size: (step + 1) * batch_size]
                    batch_labels = []
                    batch_inputs = []
                    for ids in batch:
                        int_ids_for_labels = [int(x) for x in ids]
                        int_ids_for_inputs = [int(x) for x in ids]
                        batch_labels.append(int_ids_for_labels)
                        batch_inputs.append(int_ids_for_inputs)
                    batch_labels = torch.tensor(batch_labels).long().to(device)
                    batch_inputs = torch.tensor(batch_inputs).long().to(device)

                    optimizer.zero_grad()
                    outputs = model.forward(input_ids=batch_inputs, labels=batch_labels)
                    loss, logits = outputs[:2]

                    if multi_gpu:
                        loss = loss.mean()
                    if gradient_accumulation > 1:
                        loss = loss / gradient_accumulation
                    if (step + 1) % gradient_accumulation == 0:
                        if fp16:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                        running_loss += loss.item()
                        scheduler.step()
                        optimizer.step()
                        if (step + 1) % log_step == 0:
                            print('step {} of piece {} of epoch {}, loss {}'.format(
                                (step + 1) // gradient_accumulation,
                                piece_num, epoch + 1,
                                running_loss / (log_step // gradient_accumulation)))
                            running_loss = 0
            piece_num += 1

        print('saving model for epoch {}'.format(epoch + 1))
        if not os.path.exists('./model/model_epoch{}'.format(epoch + 1)):
            os.mkdir('./model/model_epoch{}'.format(epoch + 1))
        model.save_pretrained('./model/model_epoch{}'.format(epoch + 1))
        torch.save(scheduler.state_dict(), './model/model_epoch{}/scheduler.pt'.format(epoch + 1))
        torch.save(optimizer.state_dict(), './model/model_epoch{}/optimizer.pt'.format(epoch + 1))
        print('epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('time: {}'.format(then))
        print('time for one epoch: {}'.format(then - now))

    print('training finished')
    if not os.path.exists('./model/final_model'):
        os.mkdir('./model/final_model')
    model.save_pretrained('./model/final_model')
    torch.save(scheduler.state_dict(), './model/final_model/scheduler.pt')
    torch.save(optimizer.state_dict(), './model/final_model/optimizer.pt')


if __name__ == '__main__':
    main()
