import unittest
from typing import List
import logging

logging.basicConfig(level=logging.DEBUG)

CONFIG_PATH: str = 'config/model_config_small.json'


# CONFIG_PATH: str = 'config/model_config_test.json'


class TestTrain(unittest.TestCase):
    def test_train(self):
        import sys
        from train import main
        args = [
            '--data_path', 'data/train.txt',
            '--t_total', '1000',
            '--batch_size', '2',
            # '--devices', '0'
            '--config_path', CONFIG_PATH,
            # '--config_path', 'config/model_config_test.json',
            '--epochs', '2'
        ]
        sys.argv.extend(args)
        main()
        self.assertTrue(True)


class TestLoad(unittest.TestCase):
    def test_load_from_checkpoint(self):
        import torch
        from timer import timer
        from transformers import BertTokenizer, GPT2Config, GPT2LMHeadModel, TextGenerationPipeline
        from collections import OrderedDict

        model_config_path: str = CONFIG_PATH
        vocab_path: str = 'vocab/vocab.txt'
        checkpoint_path: str = 'model/epoch=1-step=862.ckpt'

        config: GPT2Config = GPT2Config.from_json_file(model_config_path)

        # load checkpoint
        model: GPT2LMHeadModel = GPT2LMHeadModel(config=config)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        raw_state_dict: OrderedDict = checkpoint["state_dict"]
        state_dict: OrderedDict = OrderedDict({k.replace('model.', ''): v for k, v in raw_state_dict.items()})
        model.load_state_dict(state_dict)

        tokenizer: BertTokenizer = BertTokenizer(vocab_file=vocab_path)

        # model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained('model')
        # tokenizer: BertTokenizer = BertTokenizer.from_pretrained('model')

        model.eval()

        # pipeline
        with timer('pipeline'):
            text: str = '我叫'
            pipeline = TextGenerationPipeline(model, tokenizer)
            pad_token_id = tokenizer('[PAD]')['input_ids'][1]
            result = pipeline(text, max_length=20, pad_token_id=pad_token_id)
            print(result)

        # manual
        with timer('manual'):
            input_ids = tokenizer('[CLS]' + text, return_tensors='pt', padding=False, add_special_tokens=False)['input_ids']
            output_ids: torch.Tensor = model.generate(input_ids, max_length=20, pad_token_id=pad_token_id, do_sample=True)
            output_tokens: List[str] = tokenizer.convert_ids_to_tokens(output_ids[0])
            output_text: str = ''.join(filter(lambda x: x not in ['[SEP]', '[PAD]', '[CLS]'], output_tokens))
            print(output_text)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
