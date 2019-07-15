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
model = pytorch_pretrained_bert.modeling_gpt2.GPT2LMHeadModel(config=model_config)
MULTI_GPU = False
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)
    MULTI_GPU = True
model.to(device)

EPOCHS = 5
BATCH_SIZE = 4
LR = 1e-4
LR = LR * torch.cuda.device_count() if MULTI_GPU else LR
WARMUP = 0.1
LOG_STEP = 50


class CorpusDataset(object):
    def __init__(self, data_path=RAW_DATA_PATH, raw=True):
        if raw:
            with open(data_path, 'r') as f:
                print('reading lines')
                self.lines = f.readlines()
                self.all_len = len(self.lines)
            for i in tqdm(range(1000)):
                sublines = self.lines[self.all_len // 1000 * i: self.all_len // 1000 * (i + 1)]
                sublines = [full_tokenizer.tokenize(line) for line in sublines]

                sublines = [full_tokenizer.convert_tokens_to_ids(line[:n_ctx]) for line in sublines]
                sublines = pad_sequences(sublines, maxlen=n_ctx, padding='post', truncating='post')
                with open('./data/tokenized/tokenized_train_{}.txt'.format(i), 'w') as f:
                    for line in sublines:
                        for id in line[:-1]:
                            f.write(str(id) + ' ')
                        f.write(str(line[-1]))
                        f.write('\n')
            print('finish')
        else:
            self.lines = []
            for i in tqdm(range(1000)):
                with open('./data/tokenized/tokenized_train_{}.txt'.format(i), 'r') as f:
                    sub_lines = f.readlines()
                    # new_sub_lines = []
                    # for line in sub_lines:
                    #     x = line.split()[:n_ctx]
                    #     x = [int(item) for item in x]
                    #     new_sub_lines.append(x)
                    # self.lines.extend(new_sub_lines)
                    self.lines.extend([line.split()[:n_ctx] for line in sub_lines])
            self.lines = np.array(self.lines, dtype=np.int32)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]


def main():
    if raw:
        corpus_dataset = CorpusDataset(data_path=RAW_DATA_PATH, raw=raw)
        exit(1)
    total_steps = 2430 * 1000 * EPOCHS / BATCH_SIZE
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
                    batch_labels = torch.Tensor(batch_labels).long().to(device)
                    batch_inputs = torch.Tensor(batch_inputs).long().to(device)

                    optimizer.zero_grad()
                    loss = model.forward(input_ids=batch_inputs, lm_labels=batch_labels)
                    if MULTI_GPU:
                        loss.sum().backward()
                        running_loss += loss.sum().item() / torch.cuda.device_count()
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
        torch.save(model, './model.pt')

    print('training finished')
    torch.save(model, './model.pt')


if __name__ == '__main__':
    main()
