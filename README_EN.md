# GPT2-Chinese

## Description

- Chinese version of GPT2 training code, using BERT tokenizer or BPE tokenizer. from HuggingFace team [Transformers](https://github.com/huggingface/transformers). Can write poems, news, novels, or train general language models. Can write poems, news, novels, or train general language models. Support char level, word level and BPE level. training corpus.
- Chinese GPT2 training code, using BERT's Tokenizer or Sentencepiece's BPE model (thanks to [kangzhonghua](https://github.com/kangzhonghua) for the contribution, the implementation of the BPE model requires a slight modification of train.py). (Code). You can write poems, news, novels, or train common language models. Support for word units or parts of words or BPE mode (need to modify the code of train.py slightly). Supports large corpus training.

## NEWS 08.11.2020

- [CDial-GPT](https://github.com/thu-coai/CDial-GPT) (which can be loaded with this code) has been released. This project contains a rigorously cleaned Chinese dialogue dataset in a large scale liberalized domain, and a pre-trained GPT model trained on this dataset, as well as generated samples.

## NEWS 12.9.2019.

- A new project [GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat) has been released, based in part on the code of this project. It contains code for training the GPT2 conversational model with and with the training model, as well as generating samples, which you are welcome to visit.

## NEWS 12.7.2019.

- The new project [Decoders-Chinese-TF2.0](https://github.com/Morizeyao/Decoders-Chinese-TF2.0) also supports Chinese training of GPT2, which is easier to use and less likely to cause problems. It is still in the testing stage, so we welcome your comments.

## NEWS 11.9

- [GPT2-ML](https://github.com/imcaspar/gpt2-ml) (not directly related to this project) has been released, including 1.5B Chinese GPT2 model. It contains a 1.5B Chinese GPT2 model. It can be converted to the Pytorch format supported by this project for further training or test generation if you are interested.

## UPDATE 10.25

- The first pre-trained model of this project has been released, it is a prose generation model, please see the README model sharing section.

## Project Status

- When this project was announced, the Chinese GPT2 resources were almost zero, but the situation is different now. Secondly, the functionality of the project has been stabilized, so the project has been stopped for the time being. The purpose of this code is to practice using Pytorch, even if I have to fill in some holes later, there are still a lot of immature places, please understand.

## Usage

- Create a data folder in the project root directory. Put the training corpus into the data directory under the name train.json. **train.json is a json list, each element of the list is the text of an article to be trained (rather than a link to a file)**.
- Run the train.py file, check --raw, it will automatically preprocess the data.
- When the preprocessing is complete, the training will be executed automatically.

### Generate text

``` bash
python . /generate.py --length=50 --nsamples=4 --prefix=xxx --fast_pattern --save_samples --save_samples_path=/mnt/xx
```
- **--fast_pattern** (contributed by [LeeCP8](https://github.com/LeeCP8)): If the generated length parameter is relatively small, the speed is basically no difference, my personal test length = 250, faster by 2 seconds, so if you do not add--fast_pattern then fast_pattern is not used by default.
- **--save_samples**: Default is to print the output samples directly to the console, pass this parameter, it will be saved in the root directory **samples.txt**.
- **--save_samples_path**: you can specify the directory to be saved, the default is recursive creation of multi-level directories, you can not pass the file name, the default file name is **samples.txt**.

## File structure

- generate.py and train.py are generation and training scripts, respectively.
- train_single.py is an extension of train.py and can be used for a large list of individual elements (e.g. training a DouDouQiongQiong book).
- eval.py is used to evaluate the ppl score of the generated model.
- generate_texts.py is an extension of generate.py that generates several sentences starting with a list of keywords and outputs them to a file.
- train.json is an example of the format of the training samples that is available for reference.
- The cache folder contains several BERT vocabularies. make_vocab.py is a script that assists in building vocabularies on a train.json corpus file. vocab.txt is the original BERT vocabulary, vocab_all.txt is an additional archaic word, vocab_small.txt is a small vocabulary.
- The tokenizations folder contains the three tokenizers you can choose from: the default Bert Tokenizer, the split-word version of the Bert Tokenizer, and the BPE Tokenizer. 
- The scripts contain sample training and generation scripts.

## Attention.

- This project uses Bert's tokenizer to handle Chinese characters.
- If you don't use the word-splitting version of the tokenizer, you don't need to split the words yourself, the tokenizer will do it for you.
- If you use the word splitting version of the tokenizer, you should use the make_vocab.py file in the cache folder to create a word list for your corpus.
- The model needs to be calculated by yourself. If you have finished the pre-training, please feel free to talk to us.
- If your memory is very big or the corpus is small, you can change the corresponding code in the build files in train.py and preprocess the corpus without splitting it.
- If you use BPE Tokenizer, you need to build your own Chinese word list.

## Language

- It can be downloaded from [here](https://github.com/brightmart/nlp_chinese_corpus) and [here](http://thuctc.thunlp.org/#获取链接).
- The DoD language can be downloaded from [here](https://github.com/GaoPeng97/transformer-xl-chinese/tree/master/data/doupo).

## FP16 with Gradient Accumulation Support

- I've added fp16 and gradient accumulation support in the train.py file, and if you have apex installed and know what fp16 is, you can modify the variable fp16=True to enable it. But currently fp16 may not converge, for reasons unknown.

## Contact the author

- Mail: ned1991@gmail.com

## Citing

```
@misc{GPT2-Chinese,
  author = {Zeyao Du},
  title = {GPT2-Chinese: Tools for training GPT2 model in Chinese language},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Morizeyao/GPT2-Chinese}},
}
```

## Model sharing
| Model Name | Model Description | Shareholder | Link Address1 | Link Address2 | Link Address2
| The first is that the number of people in the world who have been in the hospital for more than a year has been increasing.
| Prose Model | Using 130MB of famous prose, emotional prose and prose poetry training results .  | [hughqiu](https://github.com/hughqiu "hughqiu") | [Baidu.com [fpyu](https://pan.baidu.com/s/1nbrW5iw34GRhoTin8uU2tQ) | [GDrive](https) ://drive.google.com/drive/folders/1rJC4niJKMVwixUQkuL9k5teLRnEYTmUf?usp=sharing "gDrive") |



This is the training model file of a warm and generous git user, it's open for all friends to use, and all partners are welcome to open their own training models here.


## Demo

- By user [JamesHujy](https://github.com/JamesHujy), trained on the model obtained from the code revision of this repository as a rhythm and stanza background, a new version of the [Nine Song Poetry Generator](https://jiuge.thunlp.cn/lvshi.html) is now available.
- Contributed by [leemengtaiwan](https://github.com/leemengtaiwan) to provide an [article intuitive introduction to GPT-2 and how to visualize the self-attention mechanism](https://leemeng.tw/gpt2-language-model-) generate-english-jing-yong-novels.html). Colab notebooks and models are also available (https://colab.research.google.com/drive/1MaT8-HUHfZkdCra0OqZEIr0IFCq0MJBx) for any user to generate new samples with a single click.

Translated with www.DeepL.com/Translator (free version)