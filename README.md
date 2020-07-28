# GPT2-Chinese

## Description

- Chinese version of GPT2 training code, using BERT tokenizer or BPE tokenizer. It is based on the extremely awesome repository from HuggingFace team [Transformers](https://github.com/huggingface/transformers). Can write poems, news, novels, or train general language models. Support char level, word level and BPE level. Support large training corpus.
- 中文的GPT2训练代码，使用BERT的Tokenizer或Sentencepiece的BPE model（感谢[kangzhonghua](https://github.com/kangzhonghua)的贡献，实现BPE模式需要略微修改train.py的代码）。可以写诗，新闻，小说，或是训练通用语言模型。支持字为单位或是分词模式或是BPE模式（需要略微修改train.py的代码）。支持大语料训练。

## 环境

python >= 3.6

## 注意

transformers最好使用2.11版本，鄙人试验过，最新版的transformers有错误。此外，还需要安装apex，不然也会报错[https://github.com/huggingface/transformers/issues/163]

- apex安装

    git clone https://www.github.com/nvidia/apex  
    cd apex  
    python setup.py install


