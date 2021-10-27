import tokenizers
import os
import json
import pandas as pd
from tqdm import tqdm
import re

import logging
import csv

logger = logging.getLogger(__name__)

from tokenizers.models import BPE


class CustomTokenizer:
    def __init__(self, args, dir_path, dataset_name, vocab_size, encoder_class):
        self.args = args
        self.dir_path = dir_path
        self.prefix = dataset_name
        self.vocab_size = vocab_size
        self.encoder = self.load_encoder(args, encoder_class, dir_path, dataset_name, vocab_size)

    def encode(self, text):
        return self.encoder.encode(text).ids

    def _jsonl_to_txt(self):

        data_dir = os.path.join(self.dir_path, self.prefix)
        logger.info("Flatten Corpus to text file")

        with open(os.path.join(data_dir, "train.jsonl")) as reader:
            lines = reader.readlines()
            self.data = []
            for line in lines:
                self.data.append(json.loads(line))

        df = pd.DataFrame(self.data)
        txt_file = os.path.join(data_dir, "train.txt")
        f = open(txt_file, "w")
        textlines = []

        for i, row in df.iterrows():
            # new_string = re.sub('[^a-zA-Z0-9\n\.]', ' ', row["text"])
            # new_string = re.sub(' +', ' ', new_string)
            textlines.append(row[self.args.src].replace("\n", " "))
            textlines.append(row[self.args.trg].replace("\n", " "))

        for textline in tqdm(textlines):
            f.write(textline + "\n")

    def train(self):
        self._jsonl_to_txt()
        txt_path = os.path.join(self.dir_path, self.prefix, "train.txt")
        self.encoder.train(txt_path, vocab_size=self.vocab_size, min_frequency=10)
        self.encoder.save_model(directory=self.src_dir, prefix="{0}_{1}".format(self.prefix, str(self.vocab_size)))

    def load_encoder(self, args, encoder_class, dir_path, dataset_name, vocab_size):
        self.vocab_path = args.vocab_path
        self.encoder = encoder_class()

        self.vocab_size = vocab_size
        self.prefix = dataset_name
        self.dir_path = dir_path

        self.src_dir = self.vocab_path
        base_name = os.path.join(self.src_dir, "{0}_{1}".format(self.prefix, vocab_size))
        vocab_txt_name = base_name + '-vocab.txt'
        merge_name = base_name + '-merges.txt'
        vocab_json_name = base_name + '-vocab.json'

        if os.path.exists(vocab_txt_name):
            logger.info('\ntrained encoder loaded')
            return encoder_class.from_file(vocab_txt_name)

        elif os.path.exists(merge_name):
            return encoder_class.from_file(vocab_json_name, merges_filename=merge_name)
        else:
            logger.info('\nencoder needs to be trained')
            self.train()
            return self.encoder


class CustomTEDTokenizer(CustomTokenizer):
    def __init__(self, args, dir_path, vocab_size, encoder_class, pretokenizer=None):
        super(CustomTEDTokenizer, self).__init__(args, dir_path, f"{args.src}-{args.trg}", vocab_size, encoder_class)
        self.pretokenizer = pretokenizer

    def _csv_to_txt(self):

        data_dir = os.path.join(self.dir_path, self.prefix)
        logger.info("Flatten Corpus to text file")
        dataset_path = os.path.join(self.args.root, "ted2020.tsv.gz")
        df = pd.read_csv(dataset_path, sep='\t', keep_default_na=True, encoding='utf8',
                         quoting=csv.QUOTE_NONE)
        df = df.loc[:, [self.args.src, self.args.trg]].dropna(axis=0).reset_index(drop=True)

        txt_file = os.path.join(data_dir, "for_corpus.txt")

        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        f = open(txt_file, "w")
        textlines = []

        for i, row in df.iterrows():
            src_sent = row[self.args.src].strip().replace("\n", "")
            trg_sent = row[self.args.trg].strip().replace("\n", "")

            pt = r"[.?()!@#$%^&*_+-/,]"
            # pt='[-=+,#/\?:^$.@*\"¡Ø~&%¤ý!¡»\\¡®|\(\)\[\]\<\>`\'¡¦¡·]'

            src_sent = re.sub(pt, '', src_sent)
            trg_sent = re.sub(pt, '', trg_sent)
            textlines.append(src_sent)
            textlines.append(trg_sent)
            # else:
            #     textlines.append(row[self.args.src].strip().replace("\n",""))
            #     textlines.append(row[self.args.trg].strip().replace("\n",""))

        self.textlines = textlines
        # for textline in tqdm(textlines):
        #     f.write(textline+"\n")

    def train(self):
        self._csv_to_txt()
        # txt_path = os.path.join(self.dir_path, self.prefix, "for_corpus.txt")
        # self.encoder.train(txt_path, vocab_size=self.vocab_size, min_frequency=10) # occur error

        self.encoder.train_from_iterator(self.textlines, vocab_size=self.vocab_size, min_frequency=2,
                                         special_tokens=["<unk>", "<pad>", "<eos>"],
                                         initial_alphabet=['[', '.', '?', '(', ')', '!', '@', '#', '$', '%', '^', '&',
                                                           '*', '_', '+', '-', '/', ',', ']'])

        if not os.path.isdir(self.src_dir):
            os.makedirs(self.src_dir)

        self.encoder.save_model(directory=self.src_dir, prefix=f"{self.args.src}-{self.args.trg}")

    def load_encoder(self, args, encoder_class, dir_path, dataset_name, vocab_size):
        self.vocab_path = args.vocab_path
        self.encoder = encoder_class()

        self.vocab_size = vocab_size
        self.prefix = dataset_name
        self.dir_path = dir_path

        self.src_dir = self.vocab_path
        base_name = os.path.join(self.src_dir, )
        # vocab_txt_name = base_name + '-vocab.txt'
        vocab_json_name = base_name + f"/{self.args.src}-{self.args.trg}-vocab.json"
        merge_name = base_name + f"/{self.args.src}-{self.args.trg}-merges.txt"

        print(vocab_json_name)

        if os.path.exists(merge_name):
            logger.info('\ntrained encoder loaded')
            return encoder_class.from_file(vocab_json_name, merges_filename=merge_name)
        else:
            logger.info('\nencoder needs to be trained')
            self.train()
            return self.encoder
