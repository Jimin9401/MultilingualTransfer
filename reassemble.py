from corpus_utils.bpe_mapper import Tokenizer, CustomTokenizer
from tokenizers import SentencePieceBPETokenizer, BertWordPieceTokenizer, ByteLevelBPETokenizer
from tokenizers import models
from util.args import ReassembleArgument
import logging
from transformers import AutoTokenizer, AutoConfig, RobertaTokenizer
from corpus_utils.merge import domain2pretrain, merge_domain_vocab, corpuswise_compare
from corpus_utils.tokenizer_learner import Learner
import re
import os

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    args = ReassembleArgument()
    from tokenizers.models import BPE
    encoder_class = ByteLevelBPETokenizer
    domain_tokenizer = CustomTokenizer(args=args, dir_path=args.root, encoder_class=encoder_class,
                                       dataset_name=args.dataset, vocab_size=args.vocab_size)
