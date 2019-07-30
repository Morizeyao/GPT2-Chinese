# GPT2-Chinese

## UPDATE Jul 27

- 为了保持项目的干净整洁，我做了一次更新清理。27日之前的内容以及训练斗破苍穹的脚本可以查看old_jul_27的Branch。

## 项目状态

- 目前项目处于积极开发状态，如发现任何bug或是有功能意见与改进欢迎提交Issue，PR或是联系作者。

## 使用方法

- 在项目根目录建立data文件夹。将训练语料以train.json为名放入data目录中。train.json里是一个json，json是一个列表，列表的每个元素都分别是一篇要训练的文章。
- 运行train.py文件，会自动预处理数据。
- 预处理完成之后，将train.py文件内的raw改成False，然后运行，即可开始训练。

## 文件结构

- generate.py 与 train.py 分别是生成与训练的脚本，使用pytorch-transformers库。
- cache/vocab_small.txt 与 config/model_config_small.json 是我目前试验性缩小Bert tokenizer词表大小从而缩小模型大小的产物，两者组合可以在四张2080ti上实现12的batch size。

## 注意

- pytorch-pretrained-bert现在已经改名为pytorch-transformers了，我已经将train.py更新，适应新的版本，旧版本的train.py更名为train_old.py，不再更新。
- 该训练代码可以运行，近期我会进行模型运算。
- train.txt如果太大的话内存会不足。所以我写了一个拆分若干份的代码来做预处理。
- 如果你的内存非常大或者语料较小的话，可以改掉train.py内build files内的对应代码，不做拆分直接预处理语料。
- 各位如果完成了预训练的话欢迎进行交流。

## 语料

- 可以从[这里](https://github.com/brightmart/nlp_chinese_corpus)与[这里](http://thuctc.thunlp.org/#获取链接)下载。
- 斗破苍穹语料可以从[这里](https://github.com/GaoPeng97/transformer-xl-chinese/tree/master/data/doupo)下载。

## FP16与Gradient Accumulation支持

- 我在train.py文件中加入了fp16与gradient accumulation支持，如果你安装了apex的话并且知道fp16是什么的话，可以修改变量fp61=True来启用。

## 联系作者

- QQ：330501241
- Mail：ned1991@gmail.com
 
## 生成样例

- 下为斗破苍穹的生成样例，使用约50M参数的GPT2以32Batch Size在16MB斗破苍穹小说内容上训练得到。此处[SEP]表示换行。

![avatar](sample/doupo.jpeg)

- 下为体育新闻的生成样例，使用约50M参数的GPT2以12Batch Size在约300MB体育新闻内容上训练得到。此处[SEP]表示换行，[CLS]表示新的文章，d表示数字。

![avatar](sample/tiyu.jpg)
