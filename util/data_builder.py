import wget
from .preprocessor import *
from transformers import BertTokenizer, MBart50Tokenizer
import logging
import pandas as pd
from .examples import NMTExample
import csv

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

LMAP = {"ko": "ko_KR", "fi": "fi_FI", "ja": "ja_XX", "tr": "tr_TR", "en": "en_XX", "es": "es_XX", "fr": "fr_XX",
        "ar": "ar_AR", "vi": "VI_XX"}


def load_dataset(args, tokenizer, task="nmt"):
    number_of_sample = args.n_sample
    dataset_path = os.path.join(args.root, "ted2020.tsv.gz")
    sampled_dataset_path = os.path.join(args.root, f"ted2020-all-{args.src}-{args.trg}.tsv.gz")

    cache_path = os.path.join(args.root, "cache")
    pairwise_cache = os.path.join(cache_path, f"{args.src}-{args.trg}-all")

    if args.replace_vocab:
        pairwise_cache += f"-{args.vocab_size}-replaced"
    #
    # pairwise_cache = os.path.join(cache_path, f"{args.src}-{args.trg}-all-{args.vocab_size}")

    if not os.path.isfile(sampled_dataset_path):
        df = pd.read_csv(dataset_path, sep='\t', keep_default_na=True, encoding='utf8',
                         quoting=csv.QUOTE_NONE)
        df = df.loc[:, [args.src, args.trg]].dropna(axis=0).reset_index(drop=True)
        # if args.n_sample is None:
        sampled_df = df
        # else:
        # sampled_df = df.sample(number_of_sample).reset_index(drop=True)  # sampling n examples from total data

        sampled_df.to_csv(sampled_dataset_path, index=False, encoding='utf8')

    sampled_df = pd.read_csv(sampled_dataset_path)
    all_language = sampled_df.keys()

    assert args.src in all_language and args.trg in all_language

    if not os.path.isdir(pairwise_cache):
        from sklearn.model_selection import train_test_split
        os.makedirs(pairwise_cache)

        train, test = train_test_split(sampled_df, test_size=2000, )
        train, dev = train_test_split(train, test_size=2000, )

        train.to_csv(os.path.join(pairwise_cache, "train.csv"), index=False, encoding='utf8')
        dev.to_csv(os.path.join(pairwise_cache, "dev.csv"), index=False, encoding='utf8')
        test.to_csv(os.path.join(pairwise_cache, "test.csv"), index=False, encoding='utf8')

        dev_pairs = get_pairs_from_multilingual(dev, src=args.src, trg=args.trg)
        dev_examples = convert_data_to_examples(args, tokenizer, dev_pairs, "dev", replaced=args.replace_vocab)
        pd.to_pickle(dev_examples, os.path.join(pairwise_cache, "dev.pkl"))

        test_pairs = get_pairs_from_multilingual(test, src=args.src, trg=args.trg)
        test_examples = convert_data_to_examples(args, tokenizer, test_pairs, "test", replaced=args.replace_vocab)
        pd.to_pickle(test_examples, os.path.join(pairwise_cache, "test.pkl"))

        train_pairs = get_pairs_from_multilingual(train, src=args.src, trg=args.trg)
        train_examples = convert_data_to_examples(args, tokenizer, train_pairs, "train", replaced=args.replace_vocab)
        pd.to_pickle(train_examples, os.path.join(pairwise_cache, "train.pkl"))

    else:
        train_examples = pd.read_pickle(os.path.join(pairwise_cache, "train.pkl"))
        dev_examples = pd.read_pickle(os.path.join(pairwise_cache, "dev.pkl"))
        test_examples = pd.read_pickle(os.path.join(pairwise_cache, "test.pkl"))

    return train_examples, dev_examples, test_examples


def get_pairs_from_multilingual(df: pd.DataFrame, src, trg):
    src_language = df[src].to_list()
    trg_language = df[trg].to_list()

    return {"src": src_language, "trg": trg_language}


from tqdm import tqdm
from collections import Counter


def convert_data_to_examples(args, tokenizer: MBart50Tokenizer, dataset, type="train", replaced=False):
    examples = []
    print(replaced)
    res = []
    cc = Counter()

    from tokenizers.pre_tokenizers import Whitespace
    eng_pre_tokenizer = Whitespace()

    # src_sent = " ".join([i[0] for i in eng_pre_tokenizer.pre_tokenize_str(src_sent)])

    if args.replace_vocab:
        if args.trg == "ko":
            import mecab
            pre_tokenizer = mecab.MeCab()
        elif args.trg == "ja":
            import MeCab
            pre_tokenizer = MeCab.Tagger("-Owakati")
            pre_tokenizer.morphs = pre_tokenizer.parse
        else:
            pre_tokenizer = None
    else:
        pre_tokenizer = None

    for idx, (src, trg) in tqdm(enumerate(zip(dataset["src"], dataset["trg"]))):
        if replaced:

            src = " ".join([i[0] for i in eng_pre_tokenizer.pre_tokenize_str(src)])
            src_ids = [args.new_special_src_id] + tokenizer.encode(src).ids + [2]
            if pre_tokenizer:
                if args.trg == "ko":
                    trg = " ".join(pre_tokenizer.morphs(trg))
                elif args.trg == "ja":
                    trg = " ".join(pre_tokenizer.morphs(trg).split())
                else:
                    raise NotImplementedError
            trg_ids = [args.new_special_trg_id] + tokenizer.encode(trg).ids + [2]

        else:
            src_ids = tokenizer.encode(src)
            with tokenizer.as_target_tokenizer():
                trg_ids = tokenizer.encode(trg)

        cc.update(src_ids)
        cc.update(trg_ids)

        examples.append(NMTExample(guid=f"{type}-{idx}", input_ids=src_ids, trg_ids=trg_ids))

    number_of_total = sum(cc.values())
    if args.replace_vocab:
        print(f"UNK word rate : {cc[0] / number_of_total}")
    else:
        print(f"UNK word rate : {cc[3] / number_of_total}")
    return examples


def get_ted_dataset(args, tokenizer: MBart50Tokenizer, class_of_dataset):
    DATAPATH = os.path.join("data/train-test",
                            f"{args.trg}_{args.src}")  # /home/nas1_userD/yujinbaek/dataset/text/parallel_data/ted2020

    if args.mbart:
        cache_path = os.path.join("data/train-test", "cache-mbart",
                                  f"{args.trg}_{args.src}")  # /home/nas1_userD/yujinbaek/dataset/text/parallel_data/ted2020
    else:
        cache_path = os.path.join("data/train-test", "cache-mono",
                                  f"{args.trg}_{args.src}")  # /home/nas1_userD/yujinbaek/dataset/text/parallel_data/ted2020

    train_pkl_path = os.path.join(cache_path, "train.pkl")
    dev_pkl_path = os.path.join(cache_path, "dev.pkl")
    test_pkl_path = os.path.join(cache_path, "test.pkl")

    if os.path.isdir(cache_path):

        train_dataset = pd.read_pickle(train_pkl_path)
        dev_dataset = pd.read_pickle(dev_pkl_path)
        test_dataset = pd.read_pickle(test_pkl_path)

    else:
        os.makedirs(cache_path)
        train_dataset = class_of_dataset(
            args,
            src_filename=os.path.join(DATAPATH, f'train.{args.src}'),
            tgt_filename=os.path.join(DATAPATH, f'train.{args.trg}'),
            tokenizer=tokenizer,
        ).parse("train")

        dev_dataset = class_of_dataset(
            args,
            src_filename=os.path.join(DATAPATH, f'dev.{args.src}'),
            tgt_filename=os.path.join(DATAPATH, f'dev.{args.trg}'),
            tokenizer=tokenizer,
        ).parse("dev")

        test_dataset = class_of_dataset(
            args,
            src_filename=os.path.join(DATAPATH, f'test.{args.src}'),
            tgt_filename=os.path.join(DATAPATH, f'test.{args.trg}'),
            tokenizer=tokenizer,
        ).parse("test")

        pd.to_pickle(train_dataset, train_pkl_path)
        pd.to_pickle(dev_dataset, dev_pkl_path)
        pd.to_pickle(test_dataset, test_pkl_path)

    return train_dataset, dev_dataset, test_dataset


def remove_null(dataset, src, trg):
    res = []
    dataset = dataset.loc[:, [src, trg]].dropna(axis=0).reset_index(drop=True)

    print("Remove null string")

    for idx,row in tqdm(dataset.iterrows()):
        if "NULL" in row[src]:
            continue
        if "NULL" in row[trg]:
            continue
        res.append({src:row[src],trg:row[trg]})
    print("")

    return pd.DataFrame(res)


def get_dataset(args, tokenizer):

    # train_dataset_path = os.path.join(args.root, f"all_talks_train.tsv")
    # dev_dataset_path = os.path.join(args.root, f"all_talks_dev.tsv")
    # test_dataset_path = os.path.join(args.root, f"all_talks_test.tsv")

    dataset_path = os.path.join(args.root, "ted2020.tsv.gz")
    sampled_dataset_path = os.path.join(args.root, f"ted2020-all-{args.src}-{args.trg}.tsv.gz")

    cache_path = os.path.join(args.root, "ted/cache")
    pairwise_cache = os.path.join(cache_path, f"{args.src}-{args.trg}")

    if args.mbart:
        pairwise_cache += f"-mbart"
    else:
        pairwise_cache += f"-mono"
    print(sampled_dataset_path)
    if not os.path.isfile(sampled_dataset_path):
        df = pd.read_csv(dataset_path, sep='\t', keep_default_na=True, encoding='utf8',
                         quoting=csv.QUOTE_NONE)
        df = df.loc[:, [args.src, args.trg]].dropna(axis=0).reset_index(drop=True)
        sampled_df = df
        sampled_df.to_csv(sampled_dataset_path, index=False, encoding='utf8')



    if not os.path.isdir(pairwise_cache):
        from sklearn.model_selection import train_test_split
        os.makedirs(pairwise_cache)


        train, test = train_test_split(sampled_df, test_size=2000, )
        train, dev = train_test_split(train, test_size=2000, )


        # train = pd.read_csv(train_dataset_path, sep='\t', keep_default_na=True, encoding='utf8',
        #                     quoting=csv.QUOTE_NONE)
        # train=remove_null(train,args.src,args.trg)
        #
        # dev = pd.read_csv(dev_dataset_path, sep='\t', keep_default_na=True, encoding='utf8',
        #                     quoting=csv.QUOTE_NONE)
        # dev=remove_null(dev,args.src,args.trg)
        #
        # test = pd.read_csv(test_dataset_path, sep='\t', keep_default_na=True, encoding='utf8',
        #                     quoting=csv.QUOTE_NONE)
        # test=remove_null(test,args.src,args.trg)

        train.to_csv(os.path.join(pairwise_cache, "train.csv"), index=False, encoding='utf8')
        dev.to_csv(os.path.join(pairwise_cache, "dev.csv"), index=False, encoding='utf8')
        test.to_csv(os.path.join(pairwise_cache, "test.csv"), index=False, encoding='utf8')

        dev_pairs = get_pairs_from_multilingual(dev, src=args.src, trg=args.trg)
        dev_examples = convert_data_to_examples_scratch(args, tokenizer, dev_pairs, "dev")
        pd.to_pickle(dev_examples, os.path.join(pairwise_cache, "dev.pkl"))

        test_pairs = get_pairs_from_multilingual(test, src=args.src, trg=args.trg)
        test_examples = convert_data_to_examples_scratch(args, tokenizer, test_pairs, "test")
        pd.to_pickle(test_examples, os.path.join(pairwise_cache, "test.pkl"))

        train_pairs = get_pairs_from_multilingual(train, src=args.src, trg=args.trg)
        train_examples = convert_data_to_examples_scratch(args, tokenizer, train_pairs, "train")
        pd.to_pickle(train_examples, os.path.join(pairwise_cache, "train.pkl"))

    else:
        train_examples = pd.read_pickle(os.path.join(pairwise_cache, "train.pkl"))
        dev_examples = pd.read_pickle(os.path.join(pairwise_cache, "dev.pkl"))
        test_examples = pd.read_pickle(os.path.join(pairwise_cache, "test.pkl"))

    return train_examples, dev_examples, test_examples


def convert_data_to_examples_scratch(args, tokenizer: MBart50Tokenizer, dataset, type="train"):
    examples = []
    cc = Counter()


    for idx, (src, trg) in tqdm(enumerate(zip(dataset["src"], dataset["trg"]))):

        if args.mbart:
            src_ids = tokenizer.encode(escape_attrib(src))
            with tokenizer.as_target_tokenizer():
                trg_ids = tokenizer.encode(escape_attrib(trg))
        else:
            src_ids = [len(tokenizer)] + tokenizer.encode(escape_attrib(src))[1:]
            trg_ids = [len(tokenizer)+1] + tokenizer.encode(escape_attrib(trg))[1:]
        cc.update(src_ids)
        cc.update(trg_ids)
        examples.append(NMTExample(guid=f"{type}-{idx}", input_ids=src_ids, trg_ids=trg_ids))

    return examples


def escape_attrib(s):
    s = s.replace(u"&amp;", u"&")
    s = s.replace(u"&apos;", u"'")
    s = s.replace(u"&amp;apos;", u"'")
    s = s.replace(u"&quot;", u"\"")
    s = s.replace(u"&lt;", u"<")
    s = s.replace(u"&gt;", u">")
    s = s.replace(u"&#91;", u"[")
    s = s.replace(u"&#93;", u"]")
    return s
