# GPT2-Chinese

## Description

- Chinese version of GPT2 training code, using BERT tokenizer or BPE tokenizer. Can write poems, news, novels, or train general language models. Can write poems, news, novels, or train general language models. Support char level, word level and BPE level. training corpus.
- Chinese GPT2 training code, using BERT's Tokenizer or Sentencepiece's BPE model (thanks to [kangzhonghua] for his contribution, the implementation of the BPE model requires a slight modification of the code of train.py). You can write poems, news, novels, or train common language models. Support word unit, partition mode or BPE mode (need to modify train.py's code slightly). Supports large corpus training.

## NEWS 12.7.2019

- The new project [Decoders-Chinese-TF2.0] also supports Chinese training of GPT2, which is easier to use and less likely to cause problems. It is still in the testing stage, so we welcome your comments.

## NEWS 11.9

- GPT2-ML] (not directly related to this project) has been released, including 1.5B Chinese GPT2 model. It contains a 1.5B Chinese GPT2 model. It can be converted to the Pytorch format supported by this project for further training or test generation if you are interested.

## UPDATE 10.25

- The first pre-trained model of this project has been released, it is a prose generation model, please see the README model sharing section.

## Project Status

- When this project was announced, the Chinese GPT2 resources were almost zero, but the situation is different now. Secondly, the functionality of the project has been stabilized, so the project has been stopped for the time being. The purpose of this code is to practice using Pytorch, even if I have to fill in some holes later, there are still a lot of immature places, please understand.

## Usage

- Create a data folder in the project root directory. Put the training corpus into the data directory under the name train.json. **train.json is a json list, each element of the list is the text of an article to be trained (rather than a link to a file)**.
- Run the train.py file, check --raw, it will automatically preprocess the data.
- When the preprocessing is complete, the training will be executed automatically.

### Generate text

python . /generate.py --length=50 --nsamples=4 --prefix=xxx --fast_pattern --save_samples --save_samples_path=/mnt/xx
- **--fast_pattern** (contributed by [LeeCP8]): if the generated length parameter is smaller, the speed is basically the same, I personally tested 2 seconds faster when length=250, so if you don't add--fast_pattern, then the default is not the fast_pattern method.
- **--save_samples**: By default, the output sample is printed directly to the console, passing this parameter will save it in the root directory as **samples.txt**.
- **--save_samples_path**: you can specify the directory to be saved by yourself, by default you can recursively create multi-level directories, you can't pass the file name, the default file name is **samples.txt**.

## File structure

- generate.py and train.py are generation and training scripts, respectively.
- train\_single.py is an extension of train.py, and can be used for a large list of individual elements (e.g. training a Doujian book).
- eval.py is used to evaluate the ppl score of the generated model.
- generate\_texts.py is an extension of generate.py and can be used to generate several sentences with a list of starting keywords and output them to a file.
- train.json is an example of the format of the training samples available for reference.
- The cache folder contains several BERT word lists, and make\_vocab.py is a script that assists in building the lists on a train.json corpus file. vocab.txt is the original BERT lexicon, vocab\_all.txt adds additional archaic words, and vocab\_small.txt is a small lexicon.
- In the tokenizations folder, there are three optional tokenizers, including the default Bert Tokenizer, the split-word version of the Bert Tokenizer, and the BPE Tokenizer. 
- The scripts contain sample training and generation scripts.

## Attention.

- This project uses Bert's tokenizer to handle Chinese characters.
- If you use the word splitting version of the tokenizer, you don't need to split the words yourself, the tokenizer will do it for you.
- If you use the word splitting version of the tokenizer, it is best to use the file make\\_vocab.py in the cache folder to create a word list for your corpus.
- The model needs to be calculated by yourself. If you have completed the pre-training, please feel free to talk to us.
- If your memory is very big or the corpus is small, you can change the corresponding code in the build files in train.py and pre-train the corpus without splitting it.
- If you use BPE Tokenizer, you need to build your own Chinese word list.

## Language

- It can be downloaded from [here] and [here].
- The Doom and Gloom language can be downloaded from [here].

## FP16 and Gradient Accumulation Support

- I've added fp16 and gradient accumulation support in the train.py file, and if you have apex installed and know what fp16 is, you can modify the variable fp16=True to enable it. But currently fp16 may not converge, for reasons unknown.

## Contact the author

- Mail: ned1991@gmail.com

## Citing

@misc{GPT2-Chinese,
  author = {Zeyao Du},
  title = {GPT2-Chinese: Tools for training GPT2 model in Chinese language},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Morizeyao/GPT2-Chinese}},
}

## Model sharing
| Model Name | Model Description | Shareholder | Link Address1 | Link Address2 | Link Address2
| The first is that the number of people in the world who have been in the hospital for more than a year has been increasing.
| Prose Model | Using 130MB of famous prose, emotional prose and prose poetry training results .  | The first thing you need to do is to get a copy of the book.



This is the model file from the training of a warm and generous git friend, and it is open for all friends to use.



Translated with www.DeepL.com/Translator (free version)