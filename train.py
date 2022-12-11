from transformers import GPT2LMHeadModel, GPT2Config, get_linear_schedule_with_warmup, BertTokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from typing import List
import pytorch_lightning as pl
import torch
import json
import argparse


# 11846807


class DS(Dataset):
    def __init__(self, lines: List[str], tokenizer: BertTokenizer):
        self.data = lines
        self.tok = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.data[index]
        line = self.tok.encode_plus(
            line,
            max_length=self.tok.model_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return line


class Net(pl.LightningModule):
    def __init__(
            self,
            batch_size,
            epochs,
            t_total=100000,
            config_path="config/model_config.json",
            data_path="data/train.json",
            valid_examples=100,
            vocab_path="vocab/vocab.txt",
            warm_up_steps=0,
            lr=1e-4,
    ):
        super(Net, self).__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.t_total = t_total
        self.warm_up_steps = warm_up_steps
        self.lr = lr
        self.model_name = "bert_pretrained_model"
        self.config = GPT2Config.from_json_file(config_path)
        self.config.max_length = self.config.n_positions
        self.model = GPT2LMHeadModel(config=self.config)
        self.tokenizer = BertTokenizer(vocab_file=vocab_path, model_max_length=self.config.max_length)
        with open(data_path, encoding='utf-8') as file:
            self.data = [json.loads(line.strip()) for line in file]
        self.dataset_train = DS(self.data[:-valid_examples], self.tokenizer)
        self.dataset_valid = DS(self.data[-valid_examples:], self.tokenizer)

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids
        attention_mask = attention_mask
        r = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            return_dict=True,
        )
        return r["loss"]

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_valid,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=True,
        )

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.001)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, self.warm_up_steps, self.t_total
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_nb):
        loss = self.forward(batch["input_ids"], batch["attention_mask"])

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.forward(batch["input_ids"], batch["attention_mask"])
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log(
            "val_loss",
            avg_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"val_loss": avg_loss}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--devices", default="", type=str, required=False, help="设置使用哪些显卡，用逗号分割"
    )
    parser.add_argument(
        "--accelerator", default="auto", type=str, required=False,
        help='使用的计算类型，"cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"',
        choices=["cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"]
    )
    parser.add_argument(
        "--config_path",
        default="config/model_config.json",
        type=str,
        required=False,
        help="选择模型参数",
    )
    parser.add_argument(
        "--vocab_path",
        default="vocab/vocab.txt",
        type=str,
        required=False,
        help="选择词库",
    )
    parser.add_argument(
        "--data_path",
        default="data/train.json",
        type=str,
        required=False,
        help="原始训练语料",
    )
    parser.add_argument("--epochs", default=5, type=int, required=False, help="训练循环")
    parser.add_argument(
        "--batch_size", default=8, type=int, required=False, help="训练 batch size"
    )
    parser.add_argument("--lr", default=1.5e-4, type=float, required=False, help="学习率")
    parser.add_argument(
        "--warmup_steps", default=2000, type=int, required=False, help="warm up 步数"
    )
    parser.add_argument(
        "--max_length", default=1024, type=int, required=False, help="单条文本最长长度"
    )
    parser.add_argument(
        "--eval_interval", default=100, type=int, required=False, help="eval 步数"
    )
    parser.add_argument(
        "--val_examples", default=100, type=int, required=False, help="选择多少验证集样本"
    )
    parser.add_argument(
        "--t_total", default=100000, type=int, required=False, help="计划训练多少步"
    )
    parser.add_argument(
        "--log_step", default=1, type=int, required=False, help="多少步汇报一次loss"
    )
    parser.add_argument(
        "--output_dir", default="model/", type=str, required=False, help="模型输出路径"
    )
    args = parser.parse_args()

    devices: str = args.devices
    accelerator: str = args.accelerator
    val_examples = args.val_examples
    vocab_path = args.vocab_path
    batch_size = args.batch_size
    epochs = args.epochs
    output_path = args.output_dir
    eval_interval = args.eval_interval
    lr = args.lr
    warmup_steps = args.warmup_steps
    data_path = args.data_path
    config_path = args.config_path
    t_total = args.t_total

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path,
        verbose=True,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    learning_rate_callback = LearningRateMonitor()
    trainer = pl.Trainer(
        default_root_dir=output_path,
        gradient_clip_val=1,
        max_epochs=epochs,
        devices=list(map(int, devices.split(','))) if devices else None,
        accelerator=accelerator,
        val_check_interval=eval_interval,
        callbacks=[learning_rate_callback, checkpoint_callback],
        precision=32,
    )
    net = Net(
        batch_size,
        epochs,
        t_total=t_total,
        config_path=config_path,
        data_path=data_path,
        valid_examples=val_examples,
        vocab_path=vocab_path,
        warm_up_steps=warmup_steps,
        lr=lr,
    )
    # d = torch.load('output_old/best.ckpt', map_location=torch.device("cpu"))["state_dict"]
    # d.pop('model.classifier.bias')
    # d.pop('model.classifier.weight')

    # net.load_state_dict(d, strict=False)
    trainer.fit(net)

    net.model.save_pretrained(output_path)
    net.tokenizer.save_pretrained(output_path)

    if __name__ == "__main__":
        main()
