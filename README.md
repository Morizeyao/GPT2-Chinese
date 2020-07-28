# GPT2-Chinese

## 说明

- Chinese version of GPT2 training code, using BERT tokenizer or BPE tokenizer. It is based on the extremely awesome repository from HuggingFace team [Transformers](https://github.com/huggingface/transformers). Can write poems, news, novels, or train general language models. Support char level, word level and BPE level. Support large training corpus.
- 中文的GPT2训练代码，使用BERT的Tokenizer或Sentencepiece的BPE model（感谢[kangzhonghua](https://github.com/kangzhonghua)的贡献，实现BPE模式需要略微修改train.py的代码）。可以写诗，新闻，小说，或是训练通用语言模型。支持字为单位或是分词模式或是BPE模式（需要略微修改train.py的代码）。支持大语料训练。

## 实现细节
- 每层进入之前进行layernorm
- qkv在进行自注意之前都进行升维到3 *，然后出来再降维回去
- 有一个last超参数，如果不为none，则会将上一个batch的k，v向量和当前batch的k，v在token维进行拼接，然后计算自注意力
- 英文使用bpe tokenizer，中文可以直接使用fulltokenizer，
- 位置使用1024长度
- 最后一层隐藏状态接了一个dmodel x vocabsize 的线性层，然后计算所有字的loss，一般这个loss需要除以token的数目，即平均字loss，然后反向传播。不知为何这个dmodel x vocabsize 的线性层没有和输入的词向量进行参数共享。


## 环境

python >= 3.6

## 注意

transformers最好使用2.11版本，鄙人试验过，最新版的transformers有错误。此外，还需要安装apex，不然也会报错[https://github.com/huggingface/transformers/issues/163]

- apex安装

    git clone https://www.github.com/nvidia/apex  
    cd apex  
    python setup.py install



