import pytorch_transformers
from pytorch_transformers import tokenization_bert
import torch
import numpy as np
import os
import json
import random
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
from keras.preprocessing.sequence import pad_sequences

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
full_tokenizer = tokenization_bert.BertTokenizer.from_pretrained('bert-base-chinese')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using device:', device)
model_config = pytorch_transformers.modeling_gpt2.GPT2Config.from_json_file('model_config.json')
n_ctx = model_config.n_ctx

RAW_DATA_PATH = 'data/train.txt'
tokenized_data_path = 'data/tokenized/'
raw = True  # 是否从零开始构建数据集
EPOCHS = 5
BATCH_SIZE = 4
LR = 2.5e-4
WARMUP_STEPS = 2000
LOG_STEP = 250
stride = 128
fp16 = False
fp16_opt_level = '01'
max_grad_norm = 1.0


def build_files(data_path=RAW_DATA_PATH):
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    with open(data_path, 'r') as f:
        print('reading lines')
        lines = json.load(f)
        lines = [line['c'].replace('\n', ' [SEP] ') for line in lines]  # 用[SEP]表示换行
        all_len = len(lines)
    for i in tqdm(range(1000)):
        new_lines = []
        sublines = lines[all_len // 1000 * i: all_len // 1000 * (i + 1)]
        sublines = [full_tokenizer.tokenize(line) for line in sublines if len(line) > 128]
        sublines = [full_tokenizer.convert_tokens_to_ids(line) for line in sublines]
        for subline in sublines:
            new_lines.append(subline[:n_ctx])
            start_point = stride
            while start_point + n_ctx < len(subline) + stride * 2:
                new_lines.append(subline[start_point:start_point + n_ctx])
                start_point += stride
        new_lines = pad_sequences(new_lines, maxlen=n_ctx, padding='post', truncating='post')
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'w') as f:
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

    model = pytorch_transformers.modeling_gpt2.GPT2LMHeadModel(config=model_config)
    MULTI_GPU = False
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model)
        MULTI_GPU = True
    model.to(device)

    total_lines = 0
    for i in tqdm(range(1000)):
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
            total_lines += len(f.readlines())
    total_steps = int(total_lines * EPOCHS / BATCH_SIZE)
    print('total steps = {}'.format(total_steps))
    optimizer = pytorch_transformers.AdamW(model.parameters(), lr=LR, correct_bias=True)
    scheduler = pytorch_transformers.WarmupLinearSchedule(optimizer, warmup_steps=WARMUP_STEPS,
                                                          t_total=total_steps)
    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)
    print('starting training')
    for epoch in range(EPOCHS):
        print('epoch {}'.format(epoch))
        now = datetime.now()
        print('time: {}'.format(now))
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
                    outputs = model.forward(input_ids=batch_inputs, labels=batch_labels)
                    loss, logits = outputs[:2]

                    if MULTI_GPU:
                        loss = loss.mean()

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
                    if (step + 1) % LOG_STEP == 0:
                        print('step {} of piece {} of epoch {}, loss {}'.format(step + 1, piece_num, epoch + 1,
                                                                                running_loss / LOG_STEP))
                        running_loss = 0
            piece_num += 1

        print('saving model for epoch {}'.format(epoch))
        if not os.path.exists('./model/model_epoch{}'.format(epoch)):
            os.mkdir('./model/model_epoch{}'.format(epoch))
        model.save_pretrained('./model/model_epoch{}'.format(epoch))
        torch.save(scheduler.state_dict(), './model/model_epoch{}/scheduler.pt'.format(epoch))
        torch.save(optimizer.state_dict(), './model/model_epoch{}/optimizer.pt'.format(epoch))
        print('epoch {} finished'.format(epoch))

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
