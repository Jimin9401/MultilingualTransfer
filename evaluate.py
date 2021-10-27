from util.eval_utils import *
import argparse
import os
import pandas as pd
import glob


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_name', type=str)
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


def main():
    args = get_args()
    print(os.path.basename(args.folderpath))
    df = pd.read_pickle(args.file_name)

    # df = pd.read_pickle(filename)
    predicts = df['decoded_predict'].to_list()
    predicts = [predict[2:] for predict in predicts]

    gts = df['decoded_true'].to_list()
    b = bleu_upto(gts, predicts, 5)





if __name__ == '__main__':
    main()
