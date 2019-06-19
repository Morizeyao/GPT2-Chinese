# GPT2-Chinese

## 使用方法

将训练语料以train.txt为名放入data目录中train.txt内一段文字为一行。然后运行train.py。
generate.py内包含了生成文本的代码。

## 注意

train.txt如果太大的话内存会不足。例如我的训练语料约有8个GB，那么在64GB内存的Linux机器上直接跑是跑不同的，所以写了一个拆分一千份的代码来做预处理。

## 语料

可以从[这里](https://github.com/brightmart/nlp_chinese_corpus)下载

## 联系作者

 - QQ：330501241
 - Mail：ned1991@gmail.com