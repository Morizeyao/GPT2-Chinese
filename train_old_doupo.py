import pytorch_pretrained_bert
import torch
import numpy as np
import os
import json
import re
import random
import time
from my_chinese_tokenizer import tokenization_bert
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
RAW_DATA_PATH = 'data/train_doupo.txt'
tokenized_data_path = 'data/tokenized/'
raw = True  # 是否从零开始构建数据集
full_tokenizer = tokenization_bert.BertTokenizer.from_pretrained('bert-base-chinese')
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
LR = 2.5e-4
LR = LR * torch.cuda.device_count() if MULTI_GPU else LR
WARMUP = 0.1
LOG_STEP = 50
stride = 128


def main():
    with open(RAW_DATA_PATH, 'r', encoding='utf8') as f:
        doupo = json.load(f)
        doupo = doupo.replace('\n', ' [SEP] ')
        doupo = full_tokenizer.tokenize(doupo)
        doupo = full_tokenizer.convert_tokens_to_ids(doupo)
    length = len(doupo)
    start_point = 0
    chunks = []
    while start_point < length - n_ctx:
        chunks.append(doupo[start_point: start_point + n_ctx])
        start_point += stride
    total_lines = len(chunks)

    total_steps = int(total_lines * EPOCHS / BATCH_SIZE)
    optimizer = pytorch_pretrained_bert.optimization_openai.OpenAIAdam(model.parameters(), lr=LR,
                                                                       warmup=2000 / total_steps,
                                                                       weight_decay=0.01, t_total=total_steps)
    print('starting training')
    for epoch in range(EPOCHS):
        running_loss = 0
        random.shuffle(chunks)
        for step in range(len(chunks) // BATCH_SIZE):
            batch = chunks[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]
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
            loss = model.forward(input_ids=batch_inputs, lm_labels=batch_labels)
            if MULTI_GPU:
                loss.sum().backward()
                running_loss += loss.sum().item() / torch.cuda.device_count()
            else:
                loss.backward()
                running_loss += loss.item()
            optimizer.step()
            if (step + 1) % LOG_STEP == 0:
                print('step {} of epoch {}, loss {}'.format(step + 1, epoch + 1,
                                                            running_loss / LOG_STEP))
                running_loss = 0
        print('saving model for epoch {}'.format(epoch))
        model_to_save = model.module if MULTI_GPU else model
        torch.save(model_to_save.module.state_dict(), './model.pt')

    print('training finished')
    model_to_save = model.module if MULTI_GPU else model
    torch.save(model_to_save.module.state_dict(), './model.pt')


if __name__ == '__main__':
    main()
