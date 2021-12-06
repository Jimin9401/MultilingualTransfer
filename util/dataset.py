from .examples import NMTExample
from tqdm import tqdm

from torch.utils.data import Dataset


class ParallelDataset(Dataset):
    def __init__(
            self,
            args,
            src_filename=None,
            tgt_filename=None,
            tokenizer=None,
            src_lang=None,
            tgt_lang=None,
    ):
        self.tokenizer = tokenizer
        self.args = args

        self.src_lang = src_lang if src_lang is not None else tokenizer.src_lang
        self.tgt_lang = tgt_lang if tgt_lang is not None else tokenizer.tgt_lang

        if len(self.src_lang) > 2:  # 'ko_KR' -> 'ko'
            self.src_lang = self.src_lang[:2]
        if len(self.tgt_lang) > 2:
            self.tgt_lang = self.tgt_lang[:2]

        with open(src_filename, 'r') as f:
            self.src_lines = f.read().splitlines()

        with open(tgt_filename, 'r') as f:
            self.tgt_lines = f.read().splitlines()
        self.tgt_lines = [line[6:].strip() for line in self.tgt_lines ] # remove __en__ in first

        print(f"total dataset size : {len(self.src_lines)}")

        assert len(self.src_lines) == len(self.tgt_lines)

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, index):
        src_sentence = self.src_lines[index]
        tgt_sentence = self.tgt_lines[index]
        return self.tokenize(src_sentence, tgt_sentence)

    def parse(self, name="train"):
        res = []
        for idx, (src_line, tgt_line) in tqdm(enumerate(zip(self.src_lines, self.tgt_lines))):
            examples = self.tokenize(src_line, tgt_line)
            feature = NMTExample(guid=f"{name}-{idx}", input_ids=examples["input_ids"], trg_ids=examples["labels"])
            res.append(feature)

        return res

    def tokenize(self, src_sentence, tgt_sentence):
        encoder_inputs = self.tokenizer(src_sentence, truncation=True, max_length=self.args.seq_len)
        with self.tokenizer.as_target_tokenizer():
            decoder_inputs = self.tokenizer(tgt_sentence, truncation=True, max_length=self.args.seq_len)
        inputs = {}
        inputs['input_ids'] = encoder_inputs['input_ids']
        inputs['attention_mask'] = encoder_inputs['attention_mask']
        inputs['labels'] = decoder_inputs['input_ids']
        return inputs


class TEDParallelDataset(Dataset):
    def __init__(
            self,
            args,
            src_filename,
            tgt_filename,
            tokenizer,

    ):
        # super(TEDParallelDataset, self).__init__(args,src_filename,tgt_filename,tokenizer,src_lang,tgt_lang)
        self.tokenizer = tokenizer
        self.args = args
        self.src_id = len(tokenizer.src_tokenizer)
        self.tgt_id = len(tokenizer.tgt_tokenizer)

        with open(src_filename, 'r') as f:
            self.src_lines = f.read().splitlines()
        with open(tgt_filename, 'r') as f:
            self.tgt_lines = f.read().splitlines()

        self.tgt_lines = [line[6:].strip() for line in self.tgt_lines] # to remove lang_special string

        print(f"total dataset size : {len(self.src_lines)}")
        assert len(self.src_lines) == len(self.tgt_lines)

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, index):
        src_sentence = self.src_lines[index]
        tgt_sentence = self.tgt_lines[index]
        return self.tokenize(src_sentence, tgt_sentence)

    def parse(self, name="train"):
        res = []

        for idx, (src_line, tgt_line) in tqdm(enumerate(zip(self.src_lines, self.tgt_lines))):
            examples = self.tokenize(src_line, tgt_line)
            feature = NMTExample(guid=f"{name}-{idx}", input_ids=examples["input_ids"], trg_ids=examples["labels"])
            res.append(feature)

        return res

    def tokenize(self, src_sentence, tgt_sentence):
        encoder_inputs = self.tokenizer.src_tokenizer.encode_plus(src_sentence, truncation=True,
                                                                  max_length=self.args.seq_len)
        decoder_inputs = self.tokenizer.tgt_tokenizer.encode_plus(tgt_sentence, truncation=True,
                                                                  max_length=self.args.seq_len)
        inputs = {}
        inputs['input_ids'] = [self.src_id] + encoder_inputs['input_ids'][1:]
        inputs['attention_mask'] = encoder_inputs['attention_mask']
        inputs['labels'] = [self.tgt_id] + decoder_inputs['input_ids'][1:]
        return inputs
