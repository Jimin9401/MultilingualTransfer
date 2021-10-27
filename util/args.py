import yaml
import os
import argparse


class ExperimentArgument:
    def __init__(self):

        data = {}
        parser = self.get_args()
        args = parser.parse_args()
        data.update(vars(args))
        self.data = data
        self.set_savename()
        self.__dict__ = data

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", choices=["chemprot", "rct-20k", "rct-sample", "citation_intent", "sciie", \
                                                  "ag", "hyperpartisan_news", "imdb", "amazon", "bio_ner"],
                            required=True,
                            type=str)

        parser.add_argument("--root", type=str, required=True)
        parser.add_argument("--encoder_class",
                            choices=["bert-base-uncased", "dmis-lab/biobert-base-cased-v1.1", "gpt2",
                                     "nfliu/scibert_basevocab_uncased", "allenai/scibert_scivocab_uncased"],
                            required=True, type=str)

        parser.add_argument("--n_epoch", default=10, type=int)
        parser.add_argument("--seed", default=777, type=int)
        parser.add_argument("--per_gpu_train_batch_size", default=16, type=int)
        parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int)
        parser.add_argument("--gradient_accumulation_step", default=1, type=int)
        parser.add_argument("--seq_len", default=512, type=int)
        parser.add_argument("--warmup_step", default=0, type=int)
        parser.add_argument("--decay_step", default=20000, type=int)
        parser.add_argument("--clip_norm", default=0.25, type=float)

        parser.add_argument("--weight_decay", default=0.0, type=float)
        parser.add_argument("--do_train", action="store_true")
        parser.add_argument("--do_eval", action="store_true")
        parser.add_argument("--evaluate_during_training", action="store_true")
        parser.add_argument("--init_embed", action="store_true")
        parser.add_argument("--do_test", action="store_true")
        parser.add_argument('--seed_list', nargs='+', type=int)

        parser.add_argument("--teacher_forcing_ratio", default=1.0, type=float)
        parser.add_argument("--mixed_precision", action="store_true")
        parser.add_argument("--lr", default=1e-5, type=float)
        parser.add_argument("--merge_version", action="store_true")

        parser.add_argument("--num_warmup_steps", type=int, default=0,
                            help="Number of steps for the warmup in the lr scheduler.")
        parser.add_argument("--lr_scheduler_type", default="linear", help="The scheduler type to use.",
                            choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                                     "constant_with_warmup"])

        parser.add_argument("--max_train_steps", default=None, type=int)
        parser.add_argument("--distributed_training", action="store_true")
        parser.add_argument("--checkpoint_dir", default="checkpoints", type=str)
        parser.add_argument("--test_log_dir", default="results", type=str)
        parser.add_argument("--vocab_size", type=int, default=10000)
        parser.add_argument("--use_fragment", action="store_true")
        parser.add_argument("--transfer_type", choices=["random", "average_input"], default="average_input",
                            type=str)

        # for evaluation
        parser.add_argument("--nprefix", type=int, default=50)
        parser.add_argument("--ngenerate", type=int, default=100)
        parser.add_argument("--temperature", type=float, default=1.0)
        parser.add_argument("--beam", action="store_true")
        parser.add_argument("--top_whatever", default=3, type=int)

        parser.add_argument("--decoding_strategy", choices=["likelihood", "stochastic"], default="likelihood")

        return parser

    def set_savename(self):

        if self.data["beam"] and self.data["decoding_strategy"] == "stochastic":
            raise ValueError
        if self.data["beam"] and isinstance(self.data["top_whatever"], float):
            raise ValueError

        self.data["savename"] = os.path.join(self.data["checkpoint_dir"], self.data["dataset"], str(self.data["seed"]),
                                             self.data["encoder_class"])

        if self.data["init_embed"]:
            self.data["savename"] += "_init"
            self.data["vocab_path"] = os.path.join(self.data["root"], self.data["dataset"],
                                                   self.data["encoder_class"] + "_{}".format(self.data["vocab_size"]))

        if not os.path.isdir(self.data["savename"]):
            os.makedirs(self.data["savename"])

        if self.data["do_test"]:
            self.data["model_path"] = os.path.join(self.data["checkpoint_dir"], self.data["dataset"], "{0}",
                                                   self.data["encoder_class"])
            self.data["test_dir"] = os.path.join(self.data["test_log_dir"], self.data["encoder_class"])
            self.data["test_file"] = os.path.join(self.data["test_dir"], self.data["dataset"])

            if not os.path.isdir(self.data["test_dir"]):
                os.makedirs(self.data["test_dir"])

            self.data["model_path_list"] = [self.data["model_path"].format(s) for s in self.data["seed_list"]]
            print(self.data["model_path_list"])


class ReassembleArgument:
    def __init__(self):
        data = {}
        parser = self.get_args()
        args = parser.parse_args()

        data.update(vars(args))
        self.data = data
        self.set_savename()
        self.__dict__ = data

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", choices=["wmt", "news"],
                            required=True,
                            type=str)

        parser.add_argument("--trg", choices=["ko", "ja"],
                            required=True,
                            type=str)

        parser.add_argument("--root", type=str, default="data")
        parser.add_argument("--vocab_size", type=int, default=30000)
        parser.add_argument("--target_corpus_class", choices=["monologg/bert-base-korean"],
                            default="monologg/bert-base-korean", type=str)

        return parser

    def set_savename(self):
        self.data["vocab_path"] = os.path.join(self.data["root"], self.data["dataset"],
                                               self.data["encoder_class"] + "_{}".format(self.data["vocab_size"]))

        if not os.path.isdir(self.data["vocab_path"]):
            os.makedirs(self.data["vocab_path"])


class TokenizerArgument:
    def __init__(self):
        data = {}
        parser = self.get_args()
        args = parser.parse_args()

        data.update(vars(args))
        self.data = data
        self.set_savename()
        self.__dict__ = data
        print(self.__dict__)

    def get_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--src", choices=["en", "fi"],
                            required=True,
                            type=str)

        parser.add_argument("--trg", choices=["ko", "tr", "ja", "fi"],
                            required=True,
                            type=str)

        parser.add_argument("--root", type=str, default="data")
        parser.add_argument("--vocab_size", type=int, default=50000)

        # parser.add_argument("--target_corpus_class", choices=["monologg/bert-base-korean"],
        #                     default="monologg/bert-base-korean", type=str)

        return parser

    def set_savename(self):
        self.data["vocab_path"] = os.path.join(self.data["root"],
                                               f"{self.data['src']}-{self.data['trg']}-{self.data['vocab_size']}")

        if not os.path.isdir(self.data["vocab_path"]):
            os.makedirs(self.data["vocab_path"])


class NMTArgument:
    def __init__(self):
        data = {}
        parser = self.get_args()
        args = parser.parse_args()

        data.update(vars(args))
        self.data = data
        self.set_savename()
        self.__dict__ = data

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", choices=["ted"],
                            required=True, type=str)
        parser.add_argument("--src", choices=["en", "fi"],
                            required=True, default="en", type=str)
        parser.add_argument("--trg", choices=["ko", "ja", "fi", "tr", "en"],
                            required=True,
                            type=str)
        parser.add_argument("--root", type=str, default="data")
        parser.add_argument("--n_sample", type=int, default=50000)

        parser.add_argument("--n_epoch", default=10, type=int)
        parser.add_argument("--seed", default=777, type=int)
        parser.add_argument("--per_gpu_train_batch_size", default=16, type=int)
        parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int)
        parser.add_argument("--gradient_accumulation_step", default=1, type=int)
        parser.add_argument("--seq_len", default=256, type=int)
        parser.add_argument("--warmup_step", default=0, type=int)
        parser.add_argument("--decay_step", default=20000, type=int)
        parser.add_argument("--clip_norm", default=0.25, type=float)
        parser.add_argument("--replc", default=0.25, type=float)

        parser.add_argument("--mixed_precision", action="store_true")

        parser.add_argument("--lr", default=1e-5, type=float)
        parser.add_argument("--distributed_training", action="store_true")

        parser.add_argument("--weight_decay", default=0.0, type=float)
        parser.add_argument("--do_train", action="store_true")
        parser.add_argument("--do_eval", action="store_true")
        parser.add_argument("--evaluate_during_training", action="store_true")
        parser.add_argument("--do_test", action="store_true")
        parser.add_argument("--beam", action="store_true")
        parser.add_argument("--initial_freeze", action="store_true")
        parser.add_argument("--initial_epoch_for_rearrange", type=int, default=2)

        parser.add_argument("--top_k", type=int, default=5)

        parser.add_argument("--checkpoint_dir", default="checkpoints", type=str)
        parser.add_argument("--test_dir", default="test", type=str)

        parser.add_argument("--encoder_class",
                            choices=["facebook/mbart-large-50", "facebook/mbart-50-large-many-to-many"
                                     ], default="facebook/mbart-large-50")
        parser.add_argument("--max_train_steps", default=None, type=int)
        parser.add_argument("--lr_scheduler_type", default="linear", help="The scheduler type to use.",
                            choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                                     "constant_with_warmup"])

        parser.add_argument("--num_warmup_steps", type=int, default=0,
                            help="Number of steps for the warmup in the lr scheduler.")
        parser.add_argument("--replace_vocab", action="store_true")
        parser.add_argument("--vocab_size", type=int, default=50000)
        parser.add_argument("--checkpoint_name_for_test", type=str, default="")

        return parser

    def set_savename(self):
        self.data["vocab_path"] = os.path.join(self.data["root"], self.data["dataset"],
                                               self.data["encoder_class"])
        self.data["savename"] = os.path.join(self.data["checkpoint_dir"],
                                             f"{self.data['encoder_class']}-{self.data['src']}-{self.data['trg']}")

        self.data["vocab_path"] = os.path.join(self.data["root"],
                                               f"{self.data['src']}-{self.data['trg']}-{self.data['vocab_size']}")

        if self.data["do_test"]:
            self.data["test_file"] = os.path.join(self.data["test_dir"], f"nmt-{self.data['src']}-{self.data['trg']}")
            if self.data["checkpoint_name_for_test"] == "":
                raise ValueError("Specify checkpoint name")
            if self.data["replace_vocab"]:
                self.data["test_file"] += "-replace"
        if self.data["replace_vocab"]:
            self.data["savename"] += "-replace"
