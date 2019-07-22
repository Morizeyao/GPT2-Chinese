# GPT2-Chinese

## 使用方法

- 将训练语料以train.txt为名放入data目录中train.txt内一段文字为一行。然后运行train.py。
generate.py内包含了生成文本的代码。

## 注意

- 使用该项目首先要安装pip install pytorch-pretrained-bert
- pytorch-pretrained-bert现在已经改名为pytorch-transformers了，我已经将train.py更新，适应新的版本，旧版本的train.py更名为train_old.py，不再更新。
- 该训练代码可以运行，近期我会进行模型运算。
- train.txt如果太大的话内存会不足。例如若是训练语料有8个GB，那么在64GB内存的Linux机器上直接跑是跑不通的，所以我写了一个拆分一千份的代码来做预处理。
- 如果你的内存非常大（比如你有512GB内存）或者语料较小的话，可以改掉train.py内42行之后build files内的对应代码，不做拆分直接预处理语料。
- 各位如果完成了预训练的话欢迎进行交流。

## 语料

可以从[这里](https://github.com/brightmart/nlp_chinese_corpus)下载

## FP16支持

我在train.py文件中加入了fp16支持，如果你安装了apex的话并且知道fp16是什么的话，可以修改变量fp61=True来启用。

## 联系作者

 - QQ：330501241
 - Mail：ned1991@gmail.com