from tokenizers import SentencePieceBPETokenizer
from transformers import MBart50Tokenizer


# def align_vocabularies(source_tokenizer: MBart50Tokenizer, target_tokenizer: SentencePieceBPETokenizer):
#     source_vocab = source_tokenizer.get_vocab()
#     target_vocab = target_tokenizer.get_vocab()
#
#     new_dict={}
#
#     for word, idx in target_vocab.items():
#         original_ids = [source_vocab[i] for i in source_tokenizer.tokenize(word)]
#         new_dict[idx]=original_ids
#     return new_dict


def align_vocabularies(source_tokenizer: MBart50Tokenizer, target_tokenizer: SentencePieceBPETokenizer):
    source_vocab = source_tokenizer.get_vocab()
    target_vocab = target_tokenizer.get_vocab()

    new_dict = {}

    for word, idx in target_vocab.items():
        original_ids = [source_vocab[i] for i in source_tokenizer.tokenize(word)]
        if len(original_ids) > 1 and original_ids[0] == 6:
            original_ids = original_ids[1:]
        new_dict[idx] = original_ids

    # for special tokens
    for i in range(0, 3):
        new_dict[i] = [i]
    new_dict[0] = [3]

    return new_dict
