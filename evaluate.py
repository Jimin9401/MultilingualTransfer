from util.eval_utils import *
import argparse
import os
import pandas as pd
import glob
import torch
from tokenizers import SentencePieceBPETokenizer


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_name', type=str)
    parser.add_argument('--lang', type=str)

    parser.add_argument('--str_version', action="store_true")

    return parser.parse_args()


def get_files(directory_path):
    return glob.iglob(directory_path + "*.pkl")


def remove_padding(decoded, eos_index=2):
    res = []
    for decode in decoded:
        if eos_index in decode:
            eos = decode.index(eos_index)
            decode = decode[:eos]

        res.append(decode)

    return res


import torch.nn.functional as F


# for net-based metric
def similarity(embeddings_1, embeddings_2):
    normalized_embeddings_1 = F.normalize(embeddings_1, p=2)
    normalized_embeddings_2 = F.normalize(embeddings_2, p=2)
    return torch.matmul(
        normalized_embeddings_1, normalized_embeddings_2.transpose(0, 1)
    )


def main():
    args = get_args()
    print(os.path.basename(args.folderpath))
    df = pd.read_pickle(args.file_name)

    # df = pd.read_pickle(filename)
    predicts = [sent[2:] for sent in df["decoded_predict"].to_list()]
    gts = df['decoded_true'].to_list()
    custom_tokenizer = SentencePieceBPETokenizer(f"../data/en-{args.lang}-50000/en-{args.lang}-vocab.json",
                                                 f"../data/en-{args.lang}-50000/en-{args.lang}-merges.txt")

    # reference based metric
    bleu=corpuswise_bleu(predicts,gts,)

    print(bleu)



if __name__ == '__main__':
    main()
