# GPT2-Chinese

## 使用方法

将训练语料以train.txt为名放入data目录中train.txt内一段文字为一行。然后运行train.py。
generate.py内包含了生成文本的代码。

## 注意

- 该训练代码可以运行，但是因为我本人这里算力有限且最近在做别的项目，所以没有完成预训练。各位如果有顺利完成预训练的话欢迎分享结果。
- train.txt如果太大的话内存会不足。例如我的训练语料约有8个GB，那么在64GB内存的Linux机器上直接跑是跑不同的，所以写了一个拆分一千份的代码来做预处理。
- 如果你的内存非常大（比如你有512GB内存）或者语料较小的话，可以改掉train.py内40行之后class CorpusDataset内的对应代码，不做拆分直接预处理语料。
- 因为预训练没有彻底完成，所以该repo内的generate.py代码不一定能无bug运行，但是理论上即使有bug也很容易修正，各位如果完成了预训练的话欢迎进行交流。

## 语料

可以从[这里](https://github.com/brightmart/nlp_chinese_corpus)下载

## 联系作者

 - QQ：330501241
 - Mail：ned1991@gmail.com