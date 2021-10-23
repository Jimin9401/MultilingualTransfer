
from transformers import BartTokenizer



class SharedTokenizer:
    def __init__(self, src_tokenizer:BartTokenizer, trg_tokenizer:BartTokenizer):
        self.src_tokenizer = src_tokenizer

        self.trg_tokenizer = trg_tokenizer

        self.vocab_size

    def get_overlapped_words(self):
        overlapped_words = [(k, v, self.src_tokenizer[k]) for k, v in self.trg_tokenizer.items() if
                            k in self.src_tokenizer]

        return overlapped_words