# GPT2-Chinese

## Description

- Chinese version of GPT2 training code, using BERT tokenizer. It is based on the extremely awesome repository from HuggingFace team [Transformers](https://github.com/huggingface/transformers). Can write poems, news, novels, or train general language models. Support char level, word level and BPE level. Support large training corpus.
- 中文的GPT2训练代码，使用BERT的Tokenizer。

## BIG UPDATE 04.22.2021

- 因群众普遍反映原代码对新手不够友好，且代码本身年久失修，我使用Pytorch Lightning与Transformers库重写了一版训练与预测的代码。具体使用方法请参考本README的使用方法一栏。模型本身结构与旧代码兼容。
- 这一版经过基本测试，可以完成训练与预测任务，代码在易用性上得到了提升。但是功能上会比原版代码有所缺失一些，如加载预训练模型的功能需要自己写一段小代码来添加一下。
- 原版代码保存在本项目的old_gpt_2 branch中，如有需要的话用户依然可以从中获取进行学习。
- 新版代码的依赖写在了requirements.txt文件中，请记得提前安装。
- 请注意：本项目的预测脚本直接使用的话只支持预测本项目生成的checkpoint，如果要载入huggingface官方格式的GPT2 checkpoint，请直接使用GPT2LMModel对象的from pretrained功能进行载入。之后的预测流程是一样的。

## UPDATE 02.06.2021

- 本项目新增了[通用中文GPT-2预训练模型](https://github.com/Morizeyao/GPT2-Chinese#%E6%A8%A1%E5%9E%8B%E5%88%86%E4%BA%AB)、[通用中文GPT-2预训练小模型](https://github.com/Morizeyao/GPT2-Chinese#%E6%A8%A1%E5%9E%8B%E5%88%86%E4%BA%AB)、[中文歌词GPT-2预训练模型](https://github.com/Morizeyao/GPT2-Chinese#%E6%A8%A1%E5%9E%8B%E5%88%86%E4%BA%AB)和[文言文GPT-2预训练模型](https://github.com/Morizeyao/GPT2-Chinese#%E6%A8%A1%E5%9E%8B%E5%88%86%E4%BA%AB)。模型由UER-py项目训练得到，欢迎大家使用。
此外，模型上传到了Huggingface Model Hub中。更多模型的细节请参考[gpt2-chinese-cluecorpussmall](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall)、[gpt2-distil-chinese-cluecorpussmall](https://huggingface.co/uer/gpt2-distil-chinese-cluecorpussmall)、[gpt2-chinese-lyric](https://huggingface.co/uer/gpt2-chinese-lyric)和[gpt2-chinese-ancient](https://huggingface.co/uer/gpt2-chinese-ancient)。
  
  在使用所有模型进行生成时，需要在输入的文本前加入一个起始符，如：若要输入“最美的不是下雨天，是曾与你躲过雨的屋檐”，正确的格式为“[CLS]最美的不是下雨天，是曾与你躲过雨的屋檐”。


## UPDATE 11.03.2020

- 本项目新增了[古诗词GPT-2预训练模型](https://github.com/Morizeyao/GPT2-Chinese#%E6%A8%A1%E5%9E%8B%E5%88%86%E4%BA%AB)和[对联GPT-2预训练模型](https://github.com/Morizeyao/GPT2-Chinese#%E6%A8%A1%E5%9E%8B%E5%88%86%E4%BA%AB)。模型由UER-py项目训练得到，欢迎大家使用。
此外，模型上传到了Huggingface Model Hub中。更多模型的细节请参考[gpt2-chinese-poem](https://huggingface.co/uer/gpt2-chinese-poem)和[gpt2-chinese-couplet](https://huggingface.co/uer/gpt2-chinese-couplet)。
  
  在使用古诗词模型进行生成时，需要在输入的文本前加入一个起始符，如：若要输入“梅山如积翠，”，正确的格式为“[CLS]梅山如积翠，”。
  
  对联模型训练时使用的语料格式为“上联-下联”，在使用对联模型进行生成时，需要在输入的文本前加入一个起始符，如：若要输入“丹枫江冷人初去-”，正确的格式为“[CLS]丹枫江冷人初去-”。

## NEWS 08.11.2020

- [CDial-GPT](https://github.com/thu-coai/CDial-GPT)(可用本代码载入)已发布。本项目包含一个经过严格清洗的大规模放开域中文对话数据集，本项目还包含在此数据集上训练的GPT对话预训练模型，以及生成样例，欢迎大家参观。

## NEWS 12.9.2019

- 新项目[GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat)已发布，部分基于本项目代码。包含训练GPT2对话模型的代码与与训练模型，以及生成样例，欢迎大家参观。

## NEWS 12.7.2019

- 新项目[Decoders-Chinese-TF2.0](https://github.com/Morizeyao/Decoders-Chinese-TF2.0)同样支持GPT2的中文训练，在使用上更加简单，不易产生各种问题。目前还在测试阶段，欢迎大家提出意见。

## NEWS 11.9

- [GPT2-ML](https://github.com/imcaspar/gpt2-ml)（与本项目无任何直接关联）已发布，包含1.5B中文GPT2模型。大家如有兴趣或需要可将其转换为本项目支持的Pytorch格式进行进一步训练或生成测试。

## UPDATE 10.25

- 本项目第一个预训练模型已公布，为散文生成模型，具体可查看README模型分享部分。

## 使用方法

1. 准备语料，放在data/train.json文件中，该文件的结构是：每行一个json字符串。
2. （可选）准备训练参数设置，放在config文件夹中。
3. （可选）准备tokenizer词表，放在vocab文件夹中。
4. 运行 bash train.sh进行训练。具体训练参数可以参考train.py文件中的argparse相关描述。
5. 运行 python3 generate.py 进行生成，生成的前缀可以在prefix参数中进行设置。可参考源码中参数设定部分代码。

## 注意

- 因群众普遍反映原代码对新手不够友好，且代码本身年久失修，我使用Pytorch Lightning与Transformers库重写了一版训练与预测的代码。具体使用方法请参考本README的使用方法一栏。模型本身结构与旧代码兼容。
- 这一版经过基本测试，可以完成训练与预测任务，代码在易用性上得到了提升。但是功能上会比原版代码有所缺失一些，如加载预训练模型的功能需要自己写一行代码来添加一下。
- 原版代码保存在本项目的old_gpt_2 branch中，如有需要的话用户依然可以从中获取进行学习。
- 新版代码的依赖写在了requirements.txt文件中，请记得提前安装。
- 请注意：本项目的预测脚本直接使用的话只支持预测本项目生成的checkpoint，如果要载入huggingface官方格式的GPT2 checkpoint，请直接使用GPT2LMModel对象的from pretrained功能进行载入。之后的预测流程是一样的。

## 语料

- 可以从[这里](https://github.com/brightmart/nlp_chinese_corpus)与[这里](http://thuctc.thunlp.org/#获取链接)下载。
- 斗破苍穹语料可以从[这里](https://github.com/GaoPeng97/transformer-xl-chinese/tree/master/data/doupo)下载。

## 联系作者

- Mail：ned1991@gmail.com

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

## 模型分享
|  模型名称 |   模型介绍|   分享者|  链接地址1 |  链接地址2 |
| :----------- | :----------- | :----------- | :----------- | ------------ |
| 散文模型  | 使用130MB的名家散文、情感散文和散文诗歌训练所得 。  |  [hughqiu](https://github.com/hughqiu "hughqiu") | [百度网盘【fpyu】](https://pan.baidu.com/s/1nbrW5iw34GRhoTin8uU2tQ)   | [GDrive](https://drive.google.com/drive/folders/1rJC4niJKMVwixUQkuL9k5teLRnEYTmUf?usp=sharing "GDrive") |
| 诗词模型 | 使用180MB的约80万首古诗词训练所得。 | [hhou435](https://github.com/hhou435) | [百度网盘【7fev】](https://pan.baidu.com/s/1Hy0OQ5xZcTLer9MQZW8o3g) | [GDrive](https://drive.google.com/drive/folders/1Z6nF1nrgTkrZcRLHedQHXb4_M9I7yQPN?usp=sharing) |
| 对联模型 | 使用40MB的约70万条对联训练所得。 | [hhou435](https://github.com/hhou435) | [百度网盘【i5n0】](https://pan.baidu.com/s/1j9yVQwjlXZq58wOyXK4lcg) | [GDrive](https://drive.google.com/drive/folders/1ZnsvS7oHRVueNKj_SeEhiQt86aze3ojj?usp=sharing) |
| 通用中文模型 | 使用[CLUECorpusSmall](https://github.com/CLUEbenchmark/CLUECorpus2020/)语料训练所得。 | [hhou435](https://github.com/hhou435) | [百度网盘【n3s8】](https://pan.baidu.com/s/16x0hfBCekWju75xPeyyRfA) | [GDrive](https://drive.google.com/drive/folders/1dLEANs5z4pWS0pzrak6Q2H2Nq4iYsMsf?usp=sharing) |
| 通用中文小模型 | 使用[CLUECorpusSmall](https://github.com/CLUEbenchmark/CLUECorpus2020/)语料训练所得。 | [hhou435](https://github.com/hhou435)           | [百度网盘【rpjk】](https://pan.baidu.com/s/1AiSm2GWhbGNxvhrcUlDXNA) | [GDrive](https://drive.google.com/drive/folders/1eerX1N8n_eFlnQ4xpxZ4iU2-Mx83pXFp?usp=sharing) |
| 中文歌词模型   | 使用140MB的约15万首中文歌词训练所得。                        | [hhou435](https://github.com/hhou435)           | [百度网盘【0qnn】](https://pan.baidu.com/s/19x0d0bPGCWHi9L4Pu0pSiw) | [GDrive](https://drive.google.com/drive/folders/1RFq4NoQ3phCJjrhKtu2Xbn6z0krcN9TM?usp=sharing) |
| 文言文模型 | 使用1.8GB的约300万篇文言文训练所得。 | [hhou435](https://github.com/hhou435) | [百度网盘【ek2z】](https://pan.baidu.com/s/1X3Um9HketnlGYZubY9gnew) | [GDrive](https://drive.google.com/drive/folders/1dtHTRn3fX7g8cPCCaJEXA2tmrIcImR6t?usp=sharing) |

此处为热情大方的git友训练所得的模型文件，公开给所有朋友使用，同时也欢迎各位伙伴将自己训练完毕的模型公开于此处。


## Demo

- 由用户[JamesHujy](https://github.com/JamesHujy)根据本仓库改版代码训练得到的模型作为律诗与绝句后台，新版[九歌诗歌生成器](https://jiuge.thunlp.cn/lvshi.html)已经上线。
- 由[leemengtaiwan](https://github.com/leemengtaiwan)贡献，提供[文章直觀介紹 GPT-2 以及如何視覺化自注意力機制](https://leemeng.tw/gpt2-language-model-generate-chinese-jing-yong-novels.html)。另提供 [Colab 筆記本與模型](https://colab.research.google.com/drive/1MaT8-HUHfZkdCra0OqZEIr0IFCq0MJBx)供任何使用者一鍵生成新樣例。

## 生成样例

-以下为文学散文的生成样例，由[hughqiu](https://github.com/hughqiu "hughqiu")贡献，模型已经分享于模型分享列表。语料130MB，Batch size 16，10层深度下训练10轮所得。
![avatar](sample/散文1.png)
![avatar](sample/散文2.png)
![avatar](sample/散文3.png)

- 下为斗破苍穹的生成样例，使用约50M参数的GPT2以32Batch Size在16MB斗破苍穹小说内容上训练得到。此处[SEP]表示换行。

![avatar](sample/doupo.jpeg)

- 下为古诗词的生成样例，由用户[JamesHujy](https://github.com/JamesHujy)运算并贡献。

![avatar](sample/poem_1.png)
![avatar](sample/poem_2.png)

- 下为古诗限定了生成体裁后的生成样例，由用户[JamesHujy](https://github.com/JamesHujy)运算并贡献。

![avatar](sample/律诗绝句.png)
![avatar](sample/浣溪沙_江城子.png)
![avatar](sample/蝶恋花_满江红.png)

- 下为生成剧本的样例文本，由用户[chiangandy](https://github.com/chiangandy)运算并贡献

[starttext]爱情游戏剧情讲述了钢琴父女明致怀萌的爱情、个有着努力的热情以及现实为人生的价值观众，获得一系列爱情的故事。80后录股媒体受到网友分享，是2014年主创陈拉昀出品牌总监于蓝氏集团化验师创业团门的哥哥大国度上海淮河畔，集入第一线公司青年度虽然没有放到的事业，但是蓝正是却不到位主人拒绝了解，而在蓝越的帮助理念出现，也因此开启明朗的误会而经营变成爱河。在一次偶然的编剧集电视剧之夏天上一改变了自命运环球顶樑，三人在创车祸中不知被记忆差网识分到创作，并被问流言败，以及行业服务所有的低调教同才力，陈昭和唐诗诗妍展开了一段截然不同的“2014年间段感情”，两人性格互相治癒的商业奋斗故事，尽管是共90后北京华侨大学录的一个宿舍小旅程和唐如、生等优秀青年，的人生活如何与愿违3个国偶像，并且共同创作何以此他们互相有观众的成功和关心吗?[endtext]

[starttext]学习爱情主要讲述了两对方小曼，经过啼笑皆非的考验，终于选择了三个孩子，携手共同创业来四个孩子，在大城市里创业的成功商。两家内事业的加入了北京城市，经过了一次元城市融风雨故、差异后得到异的他们，最终收获了梦想的真正属于自己的爱情。赞助理想、电视剧、剧等主创业时代人物特点在北京举行开机仪式，该剧以当下海南三个新人青年轻人面人海南梅竹马的电视角，讲述了几个在北京、喜剧代人生活中增强非浪漫的年轻人，以独特的双时代年轻人从来到北京城市化中国大城市走出发展以海南方的变迁在语种城市闯关于人生态的同时，以及他们渐渐的生活方式为自己方向上演了那么简单俗，是当代际拍摄的就如何在这个城市里都市里?那么平静的城市就是城市的风格特张嘉和支持工作打造，而这是一点就要打造出机场话剧组会。化身处处棋逢貌各种文化的人都非常独特的煽情，交织了相，滑稽等来自外衣的东北漂亮、内地，者和两位女孩子敢称是哑女孩子。交织里的人齐飞一开泰块玩笑，令人印象太趋的气质，让人眼看这个性格非常喜剧，知道的是一个“东北漂”人的外国小养家，让她耳熟练读剧的外形象显老大。之后齐飞、表示爱朗的齐飞、范儿、楚月子、白天杰。两代人的生活里友情似乎没有结合、精彩表态的开朗和丽丽丽。[endtext]

- 下為金庸武俠小說的生成樣例，由[leemengtaiwan](https://github.com/leemengtaiwan)贡献。模型大小約 82M，語料 50 MB，Batch size 16。提供[文章直觀介紹 GPT-2 以及如何視覺化自注意力機制](https://leemeng.tw/gpt2-language-model-generate-chinese-jing-yong-novels.html)。另提供 [Colab 筆記本與模型](https://colab.research.google.com/drive/1MaT8-HUHfZkdCra0OqZEIr0IFCq0MJBx)供任何使用者一鍵生成新樣例。

![avatar](sample/金庸_天龍八部.jpg)
![avatar](sample/金庸_倚天屠龍記.jpg)
![avatar](sample/金庸_鹿鼎記.jpg)
![avatar](sample/金庸_神鵰俠侶.jpg)



