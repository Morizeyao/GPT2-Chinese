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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DATA_PATH = 'data/train.txt'
tokenizer = tokenization.BasicTokenizer()
full_tokenizer = tokenization.BertTokenizer.from_pretrained('bert-base-chinese', cache_dir='./cache')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using device:', device)
model_config = pytorch_pretrained_bert.GPT2Config.from_json_file('model_config.json')
model = pytorch_pretrained_bert.modeling_gpt2.GPT2LMHeadModel(config=model_config)
MULTI_GPU = False
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)
    MULTI_GPU = True
model.to(device)

EPOCHS = 5
BATCH_SIZE = 12
LR = 2e-5
LR = LR * torch.cuda.device_count() if MULTI_GPU else LR
TRAIN_TEST_SPLIT = 0.1
WARMUP = 0.1
LOG_STEP = 250


class CorpusDataset(Dataset):
    def __init__(self, data_path=DATA_PATH, raw=True):
        if raw:
            with open(data_path, 'r') as f:
                self.lines = f.readlines()
                self.lines = tqdm([full_tokenizer.tokenize(line) for line in self.lines])
                self.lines = tqdm([full_tokenizer.convert_tokens_to_ids(line) for line in self.lines])
            with open('./data/tokenized_train.txt', 'w') as f:
                for line in self.lines:
                    f.write(line)
        else:
            with open(data_path, 'r') as f:
                self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]


def main():
    corpus_dataset = CorpusDataset()
    corpus_dataloader = DataLoader(corpus_dataset, batch_size=BATCH_SIZE, shuffle=True)

    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = pytorch_pretrained_bert.optimization.BertAdam(optimizer_grouped_parameters,
                                                              lr=LR,
                                                              warmup=WARMUP,
                                                              t_total=int(len(corpus_dataset) / BATCH_SIZE * EPOCHS))
    print('starting training')
    for epoch in range(EPOCHS):
        running_loss = 0
        epoch_data = iter(corpus_dataloader)
        for i in range(len(corpus_dataset) // BATCH_SIZE):
            batch = next(epoch_data)
            optimizer.zero_grad()
            loss = model.forward(input_ids=batch, lm_labels=batch)
            if MULTI_GPU:
                loss.sum().backward()
                running_loss += loss.sum().item() / torch.cuda.device_count()
            else:
                loss.backward()
                running_loss += loss.item()
            if (i + 1) % LOG_STEP == 0:
                print('step {} of epoch {}, loss {}'.format(i + 1, epoch + 1, running_loss / LOG_STEP))
                running_loss = 0

    print('training finished')
    torch.save(model, './model.pt')


if __name__ == '__main__':
    main()
