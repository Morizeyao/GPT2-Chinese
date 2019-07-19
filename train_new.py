import pytorch_pretrained_bert
import tokenization
import torch
import numpy as np
import os
import json
import re
import random
import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
RAW_DATA_PATH = 'data/train.txt'
tokenized_data_path = 'data/tokenized/'
raw = True  # 是否从零开始构建数据集
full_tokenizer = tokenization.BertTokenizer.from_pretrained('bert-base-chinese')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using device:', device)
model_config = pytorch_pretrained_bert.GPT2Config.from_json_file('model_config.json')
n_ctx = model_config.n_ctx
stride = 128
model = pytorch_pretrained_bert.modeling_gpt2.GPT2LMHeadModel(config=model_config)
MULTI_GPU = False
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = DataParallel(model)
    MULTI_GPU = True
model.to(device)

EPOCHS = 5
BATCH_SIZE = 4
LR = 1e-4
# LR = LR * torch.cuda.device_count() if MULTI_GPU else LR
WARMUP = 0.1
LOG_STEP = 50


def build_files(data_path=RAW_DATA_PATH, raw=True):
    with open(data_path, 'r') as f:
        print('reading lines')
        lines = json.load(f)
        lines = [line['c'] for line in lines]
        all_len = len(lines)
    for i in tqdm(range(1000)):
        new_lines = []
        sublines = lines[all_len // 1000 * i: all_len // 1000 * (i + 1)]
        sublines = [full_tokenizer.tokenize(line) for line in sublines if len(line) > 128]
        sublines = [full_tokenizer.convert_tokens_to_ids(line) for line in sublines]
        for subline in sublines:
            new_lines.append(subline[:n_ctx])
            start_point = 128
            while start_point + n_ctx < len(subline) + 256:
                new_lines.append(subline[start_point:start_point + n_ctx])
                start_point += 128
        new_lines = pad_sequences(new_lines, maxlen=n_ctx, padding='post', truncating='post')
        with open('./data/tokenized/tokenized_train_{}.txt'.format(i), 'w') as f:
            for line in new_lines:
                for id in line[:-1]:
                    f.write(str(id) + ' ')
                f.write(str(line[-1]))
                f.write('\n')
    print('finish')


def main():
    if raw:
        build_files(data_path=RAW_DATA_PATH)
        exit(1)
    total_lines = 0
    for i in tqdm(range(1000)):
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
            total_lines += len(f.readlines())
    total_steps = int(total_lines * EPOCHS / BATCH_SIZE)
    optimizer = pytorch_pretrained_bert.optimization_openai.OpenAIAdam(model.parameters(), lr=LR, warmup=0.1,
                                                                       weight_decay=0.01, t_total=total_steps)
    print('starting training')
    for epoch in range(EPOCHS):
        x = np.linspace(0, 999, 1000, dtype=np.int32)
        random.shuffle(x)
        piece_num = 0
        for i in x:
            with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
                running_loss = 0
                sub_lines = f.readlines()
                sub_lines = [line.split()[:n_ctx] for line in sub_lines]
                random.shuffle(sub_lines)
                for step in range(len(sub_lines) // BATCH_SIZE):
                    batch = sub_lines[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]
                    batch_labels = []
                    batch_inputs = []
                    for ids in batch:
                        int_ids_for_labels = [int(x) for x in ids]
                        int_ids_for_inputs = [101]
                        int_ids_for_inputs.extend([int(x) for x in ids[:-1]])  # 101 是CLS
                        batch_labels.append(int_ids_for_labels)
                        batch_inputs.append(int_ids_for_inputs)
                    batch_labels = torch.tensor(batch_labels).long().to(device)
                    batch_inputs = torch.tensor(batch_inputs).long().to(device)

                    optimizer.zero_grad()
                    loss = model.forward(input_ids=batch_inputs, lm_labels=batch_labels)
                    if MULTI_GPU:
                        loss.mean().backward()
                        running_loss += loss.mean().item() / torch.cuda.device_count()
                    else:
                        loss.backward()
                        running_loss += loss.item()
                    optimizer.step()
                    if (step + 1) % LOG_STEP == 0:
                        print('step {} of piece {} of epoch {}, loss {}'.format(step + 1, piece_num, epoch + 1,
                                                                                running_loss / LOG_STEP))
                        running_loss = 0
            piece_num += 1
        print('saving model for epoch {}'.format(epoch))
        torch.save(model.state_dict(), './model_epoch{}.pt'.format(epoch))

    print('training finished')
    torch.save(model.state_dict(), './model.pt')


if __name__ == '__main__':
    main()
