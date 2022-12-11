import unittest
from transformers import BertTokenizer, GPT2Config, GPT2LMHeadModel, TextGenerationPipeline
from collections import OrderedDict
import torch
from typing import List


class TestMain(unittest.TestCase):
    def setUp(self) -> None:
        # self.model_config_path: str = 'config/model_config_test.json'
        self.model_config_path: str = 'config/model_config_small.json'
        self.vocab_path: str = 'vocab/vocab.txt'
        self.config: GPT2Config = GPT2Config.from_json_file(self.model_config_path)

    def test_train(self):
        import sys
        from train import main
        args = [
            '--data_path', 'data/train.txt',
            '--batch_size', '2',
            # '--devices', '0'
            '--config_path', self.model_config_path,
            # '--config_path', 'config/model_config_test.json',
            '--epochs', '2'
        ]
        sys.argv.extend(args)
        main()
        self.assertTrue(True)

    @classmethod
    def pipeline(cls, model: GPT2LMHeadModel, tokenizer: BertTokenizer, text: str) -> str:
        pad_token_id = tokenizer('[PAD]')['input_ids'][1]
        pipeline = TextGenerationPipeline(model, tokenizer)
        result = pipeline(text, max_length=100, pad_token_id=pad_token_id, do_sample=True)
        return result

    @classmethod
    def generate(cls, model: GPT2LMHeadModel, tokenizer: BertTokenizer, text: str) -> str:
        pad_token_id = tokenizer('[PAD]')['input_ids'][1]
        input_ids = tokenizer('[CLS]' + text, return_tensors='pt', padding=False, add_special_tokens=False)['input_ids']
        output_ids: torch.Tensor = model.generate(input_ids, max_length=100, pad_token_id=pad_token_id, do_sample=True)
        output_tokens: List[str] = tokenizer.convert_ids_to_tokens(output_ids[0])
        output_text: str = ''.join(filter(lambda x: x not in ['[SEP]', '[PAD]', '[CLS]'], output_tokens))
        return output_text

    def load_from_checkpoint(self, checkpoint_path: str) -> GPT2LMHeadModel:
        model: GPT2LMHeadModel = GPT2LMHeadModel(config=self.config)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        raw_state_dict: OrderedDict = checkpoint["state_dict"]
        state_dict: OrderedDict = OrderedDict({k.replace('model.', ''): v for k, v in raw_state_dict.items()})
        model.load_state_dict(state_dict)
        return model

    def test_generate(self):
        # load from checkpoint
        # checkpoint_path: str = 'model/epoch=1-step=862.ckpt'
        # model = self.load_from_checkpoint(checkpoint_path)
        # tokenizer: BertTokenizer = BertTokenizer(vocab_file=self.vocab_path)

        # load using transformers api
        model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained('model/gpt2-chinese-cluecorpussmall')
        tokenizer: BertTokenizer = BertTokenizer.from_pretrained('model/gpt2-chinese-cluecorpussmall')

        model.eval()

        text: str = '我叫'

        output_text: str = self.generate(model, tokenizer, text)
        # output_text: str = self.pipeline(model, tokenizer, text)
        print(output_text)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
