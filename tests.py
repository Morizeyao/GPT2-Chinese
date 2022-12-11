import unittest
from typing import List


class TestMain(unittest.TestCase):
    def setUp(self) -> None:
        # self.model_config_path: str = 'config/model_config_test.json'
        self.model_config_path: str = 'config/model_config_small.json'

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

    def test_generate(self):
        import torch
        from transformers import BertTokenizer, GPT2Config, GPT2LMHeadModel, TextGenerationPipeline
        from collections import OrderedDict

        model_config_path: str = self.model_config_path
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

        # load using transformers api
        # model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained('model')
        # tokenizer: BertTokenizer = BertTokenizer.from_pretrained('model')

        model.eval()

        # pipeline
        text: str = '我叫'
        pipeline = TextGenerationPipeline(model, tokenizer)
        pad_token_id = tokenizer('[PAD]')['input_ids'][1]
        result = pipeline(text, max_length=20, pad_token_id=pad_token_id)
        print(result)

        # manual
        input_ids = tokenizer('[CLS]' + text, return_tensors='pt', padding=False, add_special_tokens=False)['input_ids']
        output_ids: torch.Tensor = model.generate(input_ids, max_length=20, pad_token_id=pad_token_id, do_sample=True)
        output_tokens: List[str] = tokenizer.convert_ids_to_tokens(output_ids[0])
        output_text: str = ''.join(filter(lambda x: x not in ['[SEP]', '[PAD]', '[CLS]'], output_tokens))
        print(output_text)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
