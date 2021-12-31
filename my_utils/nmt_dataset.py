import os
import random

from torch.utils.data import Dataset

def get_formatted_inputs(
    sentence,
    tokenizer=None,
    lang_id=None,
    eos_token_id=None,
    ):
    groups = sentence.split()
    token_ids = []
    # group_ids = [] ##TBD
    n_sub_tokens_in_group = []
    
    for group_id, word in enumerate(groups):
        sub_tokens = tokenizer.tokenize(word)
        token_ids.extend(tokenizer.convert_tokens_to_ids(sub_tokens))
        # group_ids.extend([group_id]*len(sub_tokens)) ##TBD
        n_sub_tokens_in_group.append(len(sub_tokens))
    
    token_ids = [lang_id] + token_ids + [eos_token_id]
    # group_ids = [-100] + group_ids + [-100] ##TBD
    return token_ids, n_sub_tokens_in_group#, group_ids


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

        if len(self.src_lang) > 2:
            self.src_lang = self.src_lang[:2]
        if len(self.tgt_lang) > 2:
            self.tgt_lang = self.tgt_lang[:2]
        
        with open(src_filename, 'r') as f:
            self.src_lines = f.read().splitlines()

        with open(tgt_filename, 'r') as f:
            self.tgt_lines = f.read().splitlines()

        assert len(self.src_lines) == len(self.tgt_lines)

    
    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, index):
        src_sentence = self.src_lines[index]
        tgt_sentence = self.tgt_lines[index]
        return self.tokenize(src_sentence, tgt_sentence)
    
    def tokenize(self, src_sentence, tgt_sentence):
        encoder_inputs = self.tokenizer(src_sentence, truncation=True, max_length=self.args.seq_len)
        with self.tokenizer.as_target_tokenizer():
            decoder_inputs = self.tokenizer(tgt_sentence, truncation=True, max_length=self.args.seq_len)
        inputs = {}
        inputs['input_ids'] = encoder_inputs['input_ids']
        inputs['attention_mask'] = encoder_inputs['attention_mask']
        inputs['labels'] = decoder_inputs['input_ids']
        return inputs

class ParallelDataset4Regularization(Dataset):
    def __init__(
        self,
        args,
        src_filename=None,
        tgt_filename=None,
        tokenizer=None,
        teacher_tokenizer=None,
        src_lang=None,
        tgt_lang=None,
    ):

        self.args = args
        self.tokenizer = tokenizer
        self.teacher_tokenizer = teacher_tokenizer

        self.src_lang = src_lang if src_lang is not None else tokenizer.src_lang
        self.tgt_lang = tgt_lang if tgt_lang is not None else tokenizer.tgt_lang

        if len(self.src_lang) > 2:
            self.src_lang = self.src_lang[:2]

        if len(self.tgt_lang) > 2:
            self.tgt_lang = self.tgt_lang[:2]
        
        with open(src_filename, 'r') as f:
            self.src_lines = f.read().splitlines()

        with open(tgt_filename, 'r') as f:
            self.tgt_lines = f.read().splitlines()

        assert len(self.src_lines) == len(self.tgt_lines)

    
    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, index):
        src_sentence = self.src_lines[index]
        tgt_sentence = self.tgt_lines[index]
        return self.tokenize(src_sentence, tgt_sentence)
    
    def tokenize(self, src_sentence, tgt_sentence):
        encoder_inputs = self.tokenizer(src_sentence, truncation=True, max_length=self.args.seq_len)
        with self.tokenizer.as_target_tokenizer():
            decoder_inputs = self.tokenizer(tgt_sentence, truncation=True, max_length=self.args.seq_len)

        if self.args.regularization_side == 'source':
            regularization_inputs = self.teacher_tokenizer(src_sentence, truncation=True, max_length=self.args.seq_len)
        elif self.args.regularization_side == 'target':
            with self.teacher_tokenizer.as_target_tokenizer():
                regularization_inputs = self.teacher_tokenizer(src_sentence, truncation=True, max_length=self.args.seq_len)
        
        inputs = {}
        inputs['input_ids'] = encoder_inputs['input_ids']
        inputs['attention_mask'] = encoder_inputs['attention_mask']
        inputs['labels'] = decoder_inputs['input_ids']
        auxiliary_inputs = {}
        auxiliary_inputs['input_ids'] = regularization_inputs['input_ids']
        auxiliary_inputs['attention_mask'] = regularization_inputs['attention_mask']

        return inputs, auxiliary_inputs

class ParallelDatasetV1(Dataset):
    # src: BERT [CLS] tok1 tok2 ... tokn [SEP]
    # tgt: skt-kogpt: tok1 tok2 ... tokn
    def __init__(
        self,
        args,
        src_filename=None,
        tgt_filename=None,
        tokenizer=None,
        src_lang=None,
        tgt_lang=None,
    ):
        self.src_tokenizer = tokenizer['src']
        self.tgt_tokenizer = tokenizer['tgt']

        self.src_lang_code = self.src_tokenizer.lang_code
        self.tgt_lang_code = self.tgt_tokenizer.lang_code
        self.args = args

        self.src_lang = src_lang if src_lang is not None else tokenizer['src'].language
        self.tgt_lang = tgt_lang if tgt_lang is not None else tokenizer['tgt'].language

        if len(self.src_lang) > 2:
            self.src_lang = self.src_lang[:2]
        if len(self.tgt_lang) > 2:
            self.tgt_lang = self.tgt_lang[:2]
        
        with open(src_filename, 'r') as f:
            self.src_lines = f.read().splitlines()

        with open(tgt_filename, 'r') as f:
            self.tgt_lines = f.read().splitlines()

        assert len(self.src_lines) == len(self.tgt_lines)

    
    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, index):
        src_sentence = self.src_lines[index]
        tgt_sentence = self.tgt_lines[index]
        return self.tokenize(src_sentence, tgt_sentence)
    
    def tokenize(self, src_sentence, tgt_sentence):
        encoder_inputs = self.src_tokenizer(src_sentence, truncation=True, max_length=self.args.seq_len)
        decoder_inputs = self.tgt_tokenizer(tgt_sentence, truncation=True, max_length=self.args.seq_len)
        inputs = {}
        inputs['input_ids'] = [self.src_lang_code] + encoder_inputs['input_ids'][1:-1] + [self.src_tokenizer.sep_token_id]
        inputs['attention_mask'] = encoder_inputs['attention_mask']
        inputs['labels'] = [self.tgt_lang_code] + decoder_inputs['input_ids'] + [self.tgt_tokenizer.eos_token_id]
        return inputs


class ParallelDatasetV2(Dataset):
    # src: skt-kogpt: tok1 tok2 ... tokn
    # tgt: BERT [CLS] tok1 tok2 ... tokn [SEP]
    def __init__(
        self,
        args,
        src_filename=None,
        tgt_filename=None,
        tokenizer=None,
        src_lang=None,
        tgt_lang=None,
    ):
        self.src_tokenizer = tokenizer['src']
        self.tgt_tokenizer = tokenizer['tgt']

        self.src_lang = src_lang if src_lang is not None else tokenizer['src'].language
        self.tgt_lang = tgt_lang if tgt_lang is not None else tokenizer['tgt'].language
        self.args = args

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        if len(self.src_lang) > 2:
            self.src_lang = self.src_lang[:2]
        if len(self.tgt_lang) > 2:
            self.tgt_lang = self.tgt_lang[:2]
        
        with open(src_filename, 'r') as f:
            self.src_lines = f.read().splitlines()

        with open(tgt_filename, 'r') as f:
            self.tgt_lines = f.read().splitlines()

        assert len(self.src_lines) == len(self.tgt_lines)

    
    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, index):
        src_sentence = self.src_lines[index]
        tgt_sentence = self.tgt_lines[index]
        return self.tokenize(src_sentence, tgt_sentence)
    
    def tokenize(self, src_sentence, tgt_sentence):
        encoder_inputs = self.src_tokenizer(src_sentence, truncation=True, max_length=self.args.seq_len)
        decoder_inputs = self.tgt_tokenizer(tgt_sentence, truncation=True, max_length=self.args.seq_len)
        inputs = {}
        inputs['input_ids'] = [self.src_lang_code] + encoder_inputs['input_ids'] + [self.src_tokenizer.eos_token_id]
        inputs['attention_mask'] = [1] + encoder_inputs['attention_mask'] + [1]
        inputs['labels'] = [self.tgt_lang_code] + decoder_inputs['input_ids']+ [self.tgt_tokenizer.sep_token_id]
        return inputs

class Dataset4EmbeddingMapping(Dataset):
    def __init__(
        self,
        args,
        src_filename=None,
        tgt_filename=None,
        tokenizer=None,
        teacher_tokenizer=None,
        src_lang=None,
        tgt_lang=None,
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.teacher_tokenizer = teacher_tokenizer

        self.src_lang = src_lang if src_lang is not None else tokenizer.src_lang
        self.tgt_lang = tgt_lang if tgt_lang is not None else tokenizer.tgt_lang

        self.src_lang_id = tokenizer.convert_tokens_to_ids(self.src_lang)
        self.tgt_lang_id = tokenizer.convert_tokens_to_ids(self.tgt_lang)
        self.eos_token_id = tokenizer.eos_token_id

        self.teacher_src_lang_id = teacher_tokenizer.convert_tokens_to_ids(self.src_lang)
        self.teacher_tgt_lang_id = teacher_tokenizer.convert_tokens_to_ids(self.tgt_lang)
        self.teacher_eos_token_id = teacher_tokenizer.eos_token_id

        if len(self.src_lang) > 2:
            self.src_lang = self.src_lang[:2]

        if len(self.tgt_lang) > 2:
            self.tgt_lang = self.tgt_lang[:2]
        
        with open(src_filename, 'r') as f:
            self.src_lines = f.read().splitlines()

        with open(tgt_filename, 'r') as f:
            self.tgt_lines = f.read().splitlines()

        assert len(self.src_lines) == len(self.tgt_lines)
    
    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, index):
        src_sentence = self.src_lines[index]
        tgt_sentence = self.tgt_lines[index]
        return self.tokenize(src_sentence, tgt_sentence)
    
    def tokenize(self, src_sentence, tgt_sentence):
        src_token_ids, n_sub_tokens_per_src_group = get_formatted_inputs(src_sentence, tokenizer=self.tokenizer, lang_id=self.src_lang_id, eos_token_id=self.eos_token_id)
        tgt_token_ids, n_sub_tokens_per_tgt_group = get_formatted_inputs(tgt_sentence, tokenizer=self.tokenizer, lang_id=self.tgt_lang_id, eos_token_id=self.eos_token_id)
        gt_src_token_ids, gt_n_sub_tokens_per_src_group = get_formatted_inputs(src_sentence, tokenizer=self.teacher_tokenizer, lang_id=self.teacher_src_lang_id, eos_token_id=self.teacher_eos_token_id)
        gt_tgt_token_ids, gt_n_sub_tokens_per_tgt_group = get_formatted_inputs(tgt_sentence, tokenizer=self.teacher_tokenizer, lang_id=self.teacher_tgt_lang_id, eos_token_id=self.teacher_eos_token_id)
        # src_token_ids, n_sub_tokens_per_src_group, src_group_ids = get_formatted_inputs(src_sentence, tokenizer=self.tokenizer, lang_id=self.src_lang_id, eos_token_id=self.eos_token_id)
        # tgt_token_ids, n_sub_tokens_per_tgt_group, tgt_group_ids = get_formatted_inputs(tgt_sentence, tokenizer=self.tokenizer, lang_id=self.tgt_lang_id, eos_token_id=self.eos_token_id)
        # gt_src_token_ids, gt_n_sub_tokens_per_src_group, gt_src_group_ids = get_formatted_inputs(src_sentence, tokenizer=self.teacher_tokenizer, lang_id=self.teacher_src_lang_id, eos_token_id=self.teacher_eos_token_id)
        # gt_tgt_token_ids, gt_n_sub_tokens_per_tgt_group, gt_tgt_group_ids = get_formatted_inputs(tgt_sentence, tokenizer=self.teacher_tokenizer, lang_id=self.teacher_tgt_lang_id, eos_token_id=self.teacher_eos_token_id)

        inputs = {}
        inputs['src_input_ids'] = src_token_ids
        inputs['tgt_input_ids'] = tgt_token_ids
        inputs['src_attention_mask'] = [1] * len(src_token_ids)
        inputs['tgt_attention_mask'] = [1] * len(tgt_token_ids)
        
        inputs['gt_src_input_ids'] = gt_src_token_ids
        inputs['gt_tgt_input_ids'] = gt_tgt_token_ids
        inputs['gt_src_attention_mask'] = [1] * len(gt_src_token_ids)
        inputs['gt_tgt_attention_mask'] = [1] * len(gt_tgt_token_ids)

        inputs['n_sub_tokens_per_src_group'] = n_sub_tokens_per_src_group
        inputs['n_sub_tokens_per_tgt_group'] = n_sub_tokens_per_tgt_group
        inputs['gt_n_sub_tokens_per_src_group'] = gt_n_sub_tokens_per_src_group
        inputs['gt_n_sub_tokens_per_tgt_group'] = gt_n_sub_tokens_per_tgt_group

        # inputs['src_group_ids'] = src_group_ids
        # inputs['tgt_group_ids'] = tgt_group_ids
        # inputs['gt_src_group_ids'] = gt_src_group_ids
        # inputs['gt_tgt_group_ids'] = gt_tgt_group_ids

        return inputs

# End-to-end version
class Dataset4End2EndAdaptation(Dataset):
    def __init__(
        self,
        args,
        src_filename=None,
        tgt_filename=None,
        tokenizer=None,
        teacher_tokenizer=None,
        src_lang=None,
        tgt_lang=None,
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.teacher_tokenizer = teacher_tokenizer

        self.src_lang = src_lang if src_lang is not None else tokenizer.src_lang
        self.tgt_lang = tgt_lang if tgt_lang is not None else tokenizer.tgt_lang

        self.src_lang_id = tokenizer.convert_tokens_to_ids(self.src_lang)
        self.tgt_lang_id = tokenizer.convert_tokens_to_ids(self.tgt_lang)
        self.eos_token_id = tokenizer.eos_token_id

        self.teacher_src_lang_id = teacher_tokenizer.convert_tokens_to_ids(self.src_lang)
        self.teacher_tgt_lang_id = teacher_tokenizer.convert_tokens_to_ids(self.tgt_lang)
        self.teacher_eos_token_id = teacher_tokenizer.eos_token_id

        if len(self.src_lang) > 2:
            self.src_lang = self.src_lang[:2]

        if len(self.tgt_lang) > 2:
            self.tgt_lang = self.tgt_lang[:2]
        
        with open(src_filename, 'r') as f:
            self.src_lines = f.read().splitlines()

        with open(tgt_filename, 'r') as f:
            self.tgt_lines = f.read().splitlines()

        assert len(self.src_lines) == len(self.tgt_lines)
    
    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, index):
        src_sentence = self.src_lines[index]
        tgt_sentence = self.tgt_lines[index]
        return self.tokenize(src_sentence, tgt_sentence)
    
    def tokenize(self, src_sentence, tgt_sentence):
        encoder_inputs = self.tokenizer(src_sentence, truncation=True, max_length=self.args.seq_len)
        with self.tokenizer.as_target_tokenizer():
            decoder_inputs = self.tokenizer(tgt_sentence, truncation=True, max_length=self.args.seq_len) # 256, 512, 1024
        inputs = {}
        inputs['input_ids'] = encoder_inputs['input_ids']
        inputs['attention_mask'] = encoder_inputs['attention_mask']
        inputs['labels'] = decoder_inputs['input_ids']

        src_token_ids, n_sub_tokens_per_src_group = get_formatted_inputs(src_sentence, tokenizer=self.tokenizer, lang_id=self.src_lang_id, eos_token_id=self.eos_token_id)
        tgt_token_ids, n_sub_tokens_per_tgt_group = get_formatted_inputs(tgt_sentence, tokenizer=self.tokenizer, lang_id=self.tgt_lang_id, eos_token_id=self.eos_token_id)
        gt_src_token_ids, gt_n_sub_tokens_per_src_group = get_formatted_inputs(src_sentence, tokenizer=self.teacher_tokenizer, lang_id=self.teacher_src_lang_id, eos_token_id=self.teacher_eos_token_id)
        gt_tgt_token_ids, gt_n_sub_tokens_per_tgt_group = get_formatted_inputs(tgt_sentence, tokenizer=self.teacher_tokenizer, lang_id=self.teacher_tgt_lang_id, eos_token_id=self.teacher_eos_token_id)
        # src_token_ids, n_sub_tokens_per_src_group, src_group_ids = get_formatted_inputs(src_sentence, tokenizer=self.tokenizer, lang_id=self.src_lang_id, eos_token_id=self.eos_token_id)
        # tgt_token_ids, n_sub_tokens_per_tgt_group, tgt_group_ids = get_formatted_inputs(tgt_sentence, tokenizer=self.tokenizer, lang_id=self.tgt_lang_id, eos_token_id=self.eos_token_id)
        # gt_src_token_ids, gt_n_sub_tokens_per_src_group, gt_src_group_ids = get_formatted_inputs(src_sentence, tokenizer=self.teacher_tokenizer, lang_id=self.teacher_src_lang_id, eos_token_id=self.teacher_eos_token_id)
        # gt_tgt_token_ids, gt_n_sub_tokens_per_tgt_group, gt_tgt_group_ids = get_formatted_inputs(tgt_sentence, tokenizer=self.teacher_tokenizer, lang_id=self.teacher_tgt_lang_id, eos_token_id=self.teacher_eos_token_id)

        inputs['src_input_ids'] = src_token_ids
        inputs['tgt_input_ids'] = tgt_token_ids
        inputs['src_attention_mask'] = [1] * len(src_token_ids)
        inputs['tgt_attention_mask'] = [1] * len(tgt_token_ids)
        
        inputs['gt_src_input_ids'] = gt_src_token_ids
        inputs['gt_tgt_input_ids'] = gt_tgt_token_ids
        inputs['gt_src_attention_mask'] = [1] * len(gt_src_token_ids)
        inputs['gt_tgt_attention_mask'] = [1] * len(gt_tgt_token_ids)

        inputs['n_sub_tokens_per_src_group'] = n_sub_tokens_per_src_group
        inputs['n_sub_tokens_per_tgt_group'] = n_sub_tokens_per_tgt_group
        inputs['gt_n_sub_tokens_per_src_group'] = gt_n_sub_tokens_per_src_group
        inputs['gt_n_sub_tokens_per_tgt_group'] = gt_n_sub_tokens_per_tgt_group

        # inputs['src_group_ids'] = src_group_ids
        # inputs['tgt_group_ids'] = tgt_group_ids
        # inputs['gt_src_group_ids'] = gt_src_group_ids
        # inputs['gt_tgt_group_ids'] = gt_tgt_group_ids

        return inputs

