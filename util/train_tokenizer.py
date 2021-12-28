import re
import torch
import tokenizers
from tokenizers import SentencePieceBPETokenizer

from transformers import PreTrainedTokenizerFast
from transformers import MBart50Tokenizer
import os

os.chdir('/home/nas1_userD/yujinbaek/code/MultilingualTransfer')
#%%
pt = r"[.?()!@#$%^&*_+-/,]"

def read_corpus(filename, pattern):
    with open(filename, 'r') as f:
        for line in f:
            yield re.sub(pattern, '', line.strip())

#dataname = 'corpora'
dataname = None

for vocab_size in [50000]:#[30000, 50000]:
    # corpus = read_corpus('/home/nas1_userD/yujinbaek/dataset/text/parallel_data/en-ko/vocab-train/corpus_to_train_vocab.txt', pt) # merged train-dev
    # corpus = read_corpus('/home/nas1_userD/yujinbaek/dataset/text/parallel_data/en-ko/vocab-train/vocab-train.ko-en.txt', pt) # train (mecab tokenized corpus)
    # corpus = read_corpus('/home/nas1_userD/yujinbaek/dataset/text/parallel_data/en-ko/vocab-train/ted-train.tok.txt', pt) # train (original corpus)
    # corpus = read_corpus('/home/nas1_userD/yujinbaek/dataset/text/parallel_data/en-ko/vocab-train/aihub_opensubtitles_tedtrain.txt', pt) # train (non-tokenized)
    # corpus = read_corpus('/home/nas1_userD/yujinbaek/dataset/text/parallel_data/en-ko/vocab-train/aihub_opensubtitles_tedtrain.tok.txt', pt) # train (non-tokenized)
    #corpus = read_corpus('/home/nas1_userD/yujinbaek/dataset/text/parallel_data/en-ko/vocab-train/vocab-train.ko-en.txt', pt) # train (non-tokenized)
    corpus = read_corpus('/home/nas1_userD/yujinbaek/dataset/text/parallel_data/en-tr/ted_train.tr-en.txt', pt)
    # corpus = read_corpus('/home/nas1_userD/yujinbaek/dataset/text/parallel_data/en-vi/vi-en/ted-train.vi-en.txt', pt)
    #corpus = read_corpus('/home/nas1_userD/yujinbaek/dataset/text/parallel_data/en-vi/vi-en/ted-train.vi-en.txt', pt)
    # corpus = read_corpus('/home/nas1_userD/yujinbaek/dataset/text/parallel_data/ted58/data/en_th/ted-train.th-en.txt', pt)
    # corpus = read_corpus('/home/nas1_userD/yujinbaek/dataset/text/parallel_data/ted58/data/en_mk/ted-train.mk-en.txt', pt)
    # corpus = read_corpus('/home/nas1_userD/yujinbaek/dataset/text/parallel_data/ted58/data/en_he/ted-train.he-en.txt', pt)
    # corpus = read_corpus('/home/nas1_userD/yujinbaek/dataset/text/parallel_data/ted58/data/en_hr/ted-train.hr-en.txt', pt)
    # corpus = read_corpus('/home/nas1_userD/yujinbaek/dataset/text/parallel_data/ted58/data/en_uk/ted-train.uk-en.txt', pt)

    min_frequency=2
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "en_XX", "tr_TR"]#, "fa_IR", "he_IL"]# "mk_MK"] #"th_TH"]# "vi_VN"]#,"ko_KR"]
    initial_alphabet=['[', '.', '?', '(', ')', '!', '@', '#', '$', '%', '^', '&',
                        # '*', '_', '+', '-', '/', ',', ']'],
                        '*', '_', '+', '-', '/', ',', ']','"',"'",","]

    # output_dir = '/home/nas1_userD/yujinbaek/dataset/text/parallel_data/en-ko/vocab-train/en-ko-shared-wo-mecab-new'
    # output_dir = '/home/nas1_userD/yujinbaek/dataset/text/parallel_data/en-th/vocab-train/en-th-shared'
    output_dir = '/home/nas1_userD/yujinbaek/dataset/text/parallel_data/en-tr/vocab-train/en-tr-shared'
    # output_dir = '/home/nas1_userD/yujinbaek/dataset/text/parallel_data/en-mk/vocab-train/en-mk-shared'
    # output_dir = '/home/nas1_userD/yujinbaek/dataset/text/parallel_data/en-he/vocab-train/en-he-shared'
    # output_dir = '/home/nas1_userD/yujinbaek/dataset/text/parallel_data/en-uk/vocab-train/en-uk-shared'

    tokenizer = SentencePieceBPETokenizer()
    tokenizer.train_from_iterator(
        corpus, 
        vocab_size=vocab_size, 
        min_frequency=min_frequency, 
        special_tokens=special_tokens, 
        initial_alphabet=initial_alphabet,
        limit_alphabet=6000, # to cover <unk> tokens
        )

    os.makedirs(output_dir, exist_ok=True)
    if dataname:
        tokenizer.save_model(output_dir, f'entr-{vocab_size}-{dataname}')
        tokenizer.save_model(output_dir, f'tren-{vocab_size}-{dataname}')
    else:
        tokenizer.save_model(output_dir, f'entr-{vocab_size}')
        tokenizer.save_model(output_dir, f'tren-{vocab_size}')
    print(f"{vocab_size} done!")