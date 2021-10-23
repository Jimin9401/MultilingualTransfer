from tokenizers import SentencePieceBPETokenizer
from transformers import MBart50Tokenizer


def align_vocabularies(source_tokenizer: MBart50Tokenizer, target_tokenizer: SentencePieceBPETokenizer):
    source_vocab = source_tokenizer.get_vocab()
    target_vocab = target_tokenizer.get_vocab()

    new_dict={}

    for word, idx in target_vocab.items():
        original_ids = [source_vocab[i] for i in source_tokenizer.tokenize(word)]
        new_dict[idx]=original_ids
    return new_dict
