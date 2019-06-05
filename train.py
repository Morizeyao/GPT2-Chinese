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
DATA_PATH = '/data/train.txt'
raw = True  # 是否从零开始构建数据集
tokenizer = tokenization.BasicTokenizer()
full_tokenizer = tokenization.BertTokenizer.from_pretrained('bert-base-chinese', cache_dir='./cache')
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
LR = 2e-5
LR = LR * torch.cuda.device_count() if MULTI_GPU else LR
TRAIN_TEST_SPLIT = 0.1
WARMUP = 0.1
LOG_STEP = 1


class CorpusDataset(object):
    def __init__(self, data_path=DATA_PATH, raw=True):
        if raw:
            with open(data_path, 'r') as f:
                print('reading lines')
                self.lines = f.readlines()
                self.all_len = len(self.lines)
            for i in tqdm(range(1000)):
                sublines = self.lines[self.all_len // 1000 * i : self.all_len // 1000 * (i + 1)]
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
            with open('./data/tokenized_train.txt', 'r') as f:
                self.lines = f.readlines()
                self.lines = [line.split()[:n_ctx] for line in self.lines]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]


def main():
    corpus_dataset = CorpusDataset(data_path=DATA_PATH, raw=raw)
    exit(1)
    optimizer = pytorch_pretrained_bert.optimization_openai.OpenAIAdam(model.parameters(), lr=LR, warmup=0.1, weight_decay=0.01)
    print('starting training')
    for epoch in range(EPOCHS):
        running_loss = 0
        for step in range(len(corpus_dataset) // BATCH_SIZE):
            batch = corpus_dataset[step * BATCH_SIZE: (step+1) * BATCH_SIZE]
            batch_labels = []
            batch_inputs = []
            for ids in batch:
                int_ids_for_labels = [int(x) for x in ids]
                int_ids_for_inputs = [101]
                int_ids_for_inputs.extend([int(x) for x in ids[:-1]]) # 101 是CLS
                batch_labels.append(int_ids_for_labels)
                batch_inputs.append(int_ids_for_inputs)
            batch_labels = torch.Tensor(batch_labels).long().to(device)
            batch_inputs = torch.Tensor(batch_inputs).long().to(device)
            print(batch_labels.shape)
            print(batch_inputs.shape)

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
                print('step {} of epoch {}, loss {}'.format(step + 1, epoch + 1, running_loss / LOG_STEP))
                running_loss = 0

    print('training finished')
    torch.save(model, './model.pt')


if __name__ == '__main__':
    main()
