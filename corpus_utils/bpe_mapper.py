import tokenizers
import os
import json
import pandas as pd
from tqdm import tqdm
import re

import logging
import csv

logger = logging.getLogger(__name__)

from contextlib import contextmanager
import os
from typing import Any, Dict, List, Optional, Tuple

from tokenizers import SentencePieceBPETokenizer

from transformers.tokenization_utils import PreTrainedTokenizer


class CustomTokenizer:
    def __init__(self, args, dir_path, dataset_name, vocab_size, encoder_class, pretokenizer):
        self.args = args
        self.dir_path = dir_path
        self.prefix = dataset_name
        self.vocab_size = vocab_size
        self.pretokenizer = pretokenizer
        from tokenizers.pre_tokenizers import Whitespace
        self.eng_pre_tokenizer = Whitespace()

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
        super(CustomTEDTokenizer, self).__init__(args, dir_path, f"{args.src}-{args.trg}", vocab_size, encoder_class,
                                                 pretokenizer)
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
            src_sent = row[self.args.src].replace("\n", "").strip()
            trg_sent = row[self.args.trg].replace("\n", "").strip()

            # pt = r"[.?()!@#$%^&*_+-/,]"
            # pt='[-=+,#/\?:^$.@*\"��~&%��!��\\��|\(\)\[\]\<\>`\'����]'

            # src_sent = re.sub(pt, '', src_sent)
            # trg_sent = re.sub(pt, '', trg_sent)

            src_sent = " ".join([i[0] for i in self.eng_pre_tokenizer.pre_tokenize_str(src_sent)])

            textlines.append(src_sent)

            if self.pretokenizer:
                if self.args.trg == "ko":
                    trg_sent = " ".join(self.pretokenizer.morphs(trg_sent))
                elif self.args.trg == "ja":
                    trg_sent = " ".join(self.pretokenizer.morphs(trg_sent).split())

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
                                                           '*', '_', '+', '-', '/', ',', ']', '"', "'", ","])

        # self.encoder.train_from_iterator(self.textlines, vocab_size=self.vocab_size, min_frequency=2,
        #                                  special_tokens=["<unk>", "<pad>", "<eos>"],
        #                                  initial_alphabet=['[', '.', '?', '(', ')', '!', '@', '#', '$', '%', '^', '&',
        #                                                    '*', '_', '+', '-', '/', ',', ']'])

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


class NMTTokenizer(PreTrainedTokenizer):
    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

    def __init__(
            self,
            vocab_filename,
            merges_filename,
            src_lang=None,
            tgt_lang=None,
            bos_token="<s>",
            eos_token="</s>",
            sep_token="</s>",
            cls_token="<s>",
            unk_token="<unk>",
            pad_token="<pad>",
            additional_special_tokens=None,
            **kwargs
    ):
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
        )

        self.bpe_model = SentencePieceBPETokenizer.from_file(
            vocab_filename,
            merges_filename,
        )

        self.vocab = self.bpe_model.get_vocab()
        self.id2token = {v: k for (k, v) in self.vocab.items()}

        self._src_lang = src_lang if src_lang is not None else "en_XX"
        self.cur_lang_code_id = self.vocab[self._src_lang]
        self.tgt_lang = tgt_lang
        self.set_src_lang_special_tokens(self._src_lang)

    def get_vocab(self):
        return self.vocab

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def src_lang(self) -> str:
        return self._src_lang

    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)

    def _tokenize(self, text: str) -> List[str]:
        return self.bpe_model.encode(text).tokens

    def _convert_token_to_id(self, token):
        return self.vocab[token]

    def _convert_id_to_token(self, index):
        return self.id2token[index]

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:

        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    def _build_translation_inputs(
            self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this bpe_model")
        self.src_lang = src_lang
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
        inputs["forced_bos_token_ids"] = tgt_lang_id
        return inputs

    def prepare_seq2seq_batch(
            self,
            src_texts: List[str],
            src_lang: str = "en_XX",
            tgt_texts: Optional[List[str]] = None,
            tgt_lang: str = "ko_KR",
            **kwargs,
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    @contextmanager
    def as_target_tokenizer(self):
        self.set_tgt_lang_special_tokens(self.tgt_lang)
        yield
        self.set_src_lang_special_tokens(self.src_lang)

    def set_src_lang_special_tokens(self, src_lang) -> None:
        self.cur_lang_code = self.vocab[src_lang]
        self.prefix_tokens = [self.cur_lang_code]
        self.suffix_tokens = [self.eos_token_id]

    def set_tgt_lang_special_tokens(self, tgt_lang) -> None:
        self.cur_lang_code = self.vocab[tgt_lang]
        self.prefix_tokens = [self.cur_lang_code]
        self.suffix_tokens = [self.eos_token_id]

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace("▁", " ").strip()
        return out_string

    def _decode(
            self,
            token_ids: List[int],
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = True,
            spaces_between_special_tokens: bool = True,
            **kwargs
    ) -> str:

        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separately for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            elif token in ['vi_VN', 'en_XX', 'ko_KR', 'th_TH', 'mk_MK', 'he_IL', 'hr_HR', 'uk_UA', 'fa_IR']:
                continue
            text.append(token)
        if text:
            text = self.convert_tokens_to_string(text)
        else:
            text = " "
        return text


class CustomNMTTokenizer(PreTrainedTokenizer):
    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

    def __init__(
            self,
            src_tokenizer,
            tgt_tokenizer,
            src_lang=None,
            tgt_lang=None,
            bos_token="<s>",
            eos_token="</s>",
            sep_token="</s>",
            cls_token="<s>",
            unk_token="<unk>",
            pad_token="<pad>",
            additional_special_tokens=None,
            **kwargs
    ):
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
        )

        self.src_tokenizer=src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.bpe_model = src_tokenizer

        self.vocab = self.bpe_model.get_vocab()
        self.id2token = {v: k for (k, v) in self.vocab.items()}

        self._src_lang = src_lang if src_lang is not None else "en_XX"
        # self.cur_lang_code_id = self.vocab[self._src_lang]
        # self.tgt_lang = tgt_lang
        # self.set_src_lang_special_tokens(self._src_lang)

    def get_vocab(self):
        return self.vocab

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def src_lang(self) -> str:
        return self._src_lang

    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)

    def _tokenize(self, text: str) -> List[str]:
        return self.bpe_model.encode(text)

    def _convert_token_to_id(self, token):
        return self.vocab[token]

    def _convert_id_to_token(self, index):
        return self.id2token[index]

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:

        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    def _build_translation_inputs(
            self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this bpe_model")
        self.src_lang = src_lang
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
        inputs["forced_bos_token_ids"] = tgt_lang_id
        return inputs

    def prepare_seq2seq_batch(
            self,
            src_texts: List[str],
            src_lang: str = "en_XX",
            tgt_texts: Optional[List[str]] = None,
            tgt_lang: str = "ko_KR",
            **kwargs,
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    @contextmanager
    def as_target_tokenizer(self):
        self.bpe_model=self.tgt_tokenizer

    @contextmanager
    def as_source_tokenizer(self):
        self.bpe_model=self.src_tokenizer

    def set_src_lang_special_tokens(self, src_lang) -> None:
        self.cur_lang_code = self.vocab[src_lang]
        self.prefix_tokens = [self.cur_lang_code]
        self.suffix_tokens = [self.eos_token_id]

    def set_tgt_lang_special_tokens(self, tgt_lang) -> None:
        self.cur_lang_code = self.vocab[tgt_lang]
        self.prefix_tokens = [self.cur_lang_code]
        self.suffix_tokens = [self.eos_token_id]

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace("▁", " ").strip()
        return out_string

    def _decode(
            self,
            token_ids: List[int],
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = True,
            spaces_between_special_tokens: bool = True,
            **kwargs
    ) -> str:

        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separately for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            elif token in ['vi_VN', 'en_XX', 'ko_KR', 'th_TH', 'mk_MK', 'he_IL', 'hr_HR', 'uk_UA', 'fa_IR']:
                continue
            text.append(token)
        if text:
            text = self.convert_tokens_to_string(text)
        else:
            text = " "
        return text
