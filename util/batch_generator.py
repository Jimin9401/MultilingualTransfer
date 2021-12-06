import pandas as pd
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, Dataset
import math
from tqdm import tqdm
import random

from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional, Union
from .examples import NMTExample,CFExample

#
# class Dataset(Dataset):
#     def __init__(self, X, y):
#         """Reads source and target sequences from txt files."""
#         self.X = X
#         self.y = y
#
#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, index):
#         """Returns one data pair (source and target)."""
#         data = {}
#         data["X"] = self.X[index]
#         data["y"] = self.y[index]
#         return data


class Base_Batchfier(IterableDataset):
    def __init__(self, args, df,batch_size: int = 32, maxlen: int = 512, padding_index=70000, device="cuda"):
        super(Base_Batchfier).__init__()
        self.df=df
        self.args = args
        self.maxlen = maxlen
        self.size = batch_size
        self.padding_index = padding_index
        self.device = device
        # self.epoch_shuffle = epoch_shuffle
        # self.size = len(self.df) / num_buckets

    # def truncate_small(self, df, criteria='lens'):
    #     lens = np.array(df[criteria])
    #     indices = np.nonzero((lens < self.minlen).astype(np.int64))[0]
    #     return df.drop(indices)
    #
    # def truncate_large(self, texts, lens):
    #     new_texts = []
    #     new_lens = []
    #     for i in range(len(texts)):
    #         text = texts[i]
    #         if len(text) > self.maxlen:
    #             new_texts.append(text[:self.maxlen])
    #             new_lens.append(self.maxlen)
    #         else:
    #             remainder = len(text) % self.seq_len
    #             l = lens[i]
    #             if remainder and remainder < 10:
    #                 text = text[:-remainder]
    #                 l = l - remainder
    #             new_texts.append(text)
    #             new_lens.append(l)
    #     return new_texts, new_lens

    def shuffle(self, df, num_buckets):
        dfs = []
        for bucket in range(num_buckets - 1):
            new_df = df.iloc[bucket * self.size: (bucket + 1) * self.size]
            dfs.append(new_df)
        random.shuffle(dfs)
        dfs.append(df.iloc[num_buckets - 1 * self.size: num_buckets * self.size])
        df = pd.concat(dfs)
        return df







class NMTBatchfier(Base_Batchfier):
    def __init__(self, args, df: List[NMTExample], batch_size: int = 32, maxlen: int = 512,
                 padding_index=1, device="cuda"):
        super(NMTBatchfier, self).__init__(args, df, batch_size, maxlen, padding_index,device)
        self.num_buckets = len(self.df) // self.size + (len(self.df) % self.size != 0)
        self.df = sorted(self.df, key=lambda x: (len(x.input_ids), len(x.trg_ids)), reverse=True)

        print(f"total dataset size : {len(self.df)}")

    def __iter__(self):
        for example in self.df:
            src = example.input_ids[:self.maxlen]
            trg = example.trg_ids[:self.maxlen]

            yield src, trg
            # yield trg, src

    def __len__(self):
        return self.num_buckets

    def collate(self, batch):
        src_ids = [torch.LongTensor(item[0]) for item in batch]
        trg_ids = [torch.LongTensor(item[1]) for item in batch]

        src_ids = pad_sequence(src_ids, batch_first=True, padding_value=self.padding_index)
        trg_ids = pad_sequence(trg_ids, batch_first=True, padding_value=self.padding_index)

        labels = trg_ids[:, 1:]
        trg_ids = trg_ids[:, :-1]

        src_attn = (src_ids != self.padding_index).to(torch.long)
        trg_attn = (trg_ids != self.padding_index).to(torch.long)

        return {"input_ids": src_ids.to(self.device), "attention_mask": src_attn.to(self.device),
                "decoder_input_ids": trg_ids.to(self.device),
                "decoder_attention_mask": trg_attn.to(self.device), "labels": labels.to(self.device)}


class CFBatchfier(Base_Batchfier):
    def __init__(self, args, df: List[CFExample], batch_size: int = 32, maxlen: int = 512,
                 padding_index=1, device="cuda"):
        super(CFBatchfier, self).__init__(args, df, batch_size, maxlen, padding_index,device)
        self.num_buckets = len(self.df) // self.size + (len(self.df) % self.size != 0)

    def __iter__(self):
        for example in self.df:
            src = example.input_ids[:self.maxlen]
            trg = example.category_id[:self.maxlen]
            yield src, trg

    def __len__(self):
        return self.num_buckets

    def collate(self, batch):
        src_ids = [torch.LongTensor(item[0]) for item in batch]
        trg_ids = [torch.LongTensor(item[1]) for item in batch]

        src_ids = pad_sequence(src_ids, batch_first=True, padding_value=self.padding_index)
        trg_ids = pad_sequence(trg_ids, batch_first=True, padding_value=self.padding_index)

        labels = trg_ids[:, 1:]
        trg_ids = trg_ids[:, :-1]

        src_attn = (src_ids != self.padding_index).to(torch.long)
        trg_attn = (trg_ids != self.padding_index).to(torch.long)

        return {"input_ids": src_ids.to(self.device), "attention_mask": src_attn.to(self.device),
                "decoder_input_ids": trg_ids.to(self.device),
                "decoder_attention_mask": trg_attn.to(self.device), "labels": labels.to(self.device)}
