from util.args import TokenizerArgument
import logging
from tokenizers import SentencePieceBPETokenizer
from corpus_utils.bpe_mapper import CustomTokenizer,CustomTEDTokenizer


VocabMap = {"en": "bert-base-uncased", "ko": "monologg/koelectra-base-v3-discriminator"}


logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    args = TokenizerArgument()
    # src_lang_path = VocabMap[args.src]
    # trg_lang_path = VocabMap[args.trg]

    encoder_class = SentencePieceBPETokenizer
    domain_tokenizer = CustomTEDTokenizer(args=args, dir_path=args.root, encoder_class=encoder_class,
                                       vocab_size=args.vocab_size)

    tokenizer=CustomTEDTokenizer(args=args, dir_path=args.root, encoder_class=encoder_class,
                       vocab_size=args.vocab_size).encoder
    print(tokenizer)

# if __name__ == "__main__":
#     args = TokenizerArgument()
#     src_lang_path = VocabMap[args.src]
#     trg_lang_path = VocabMap[args.trg]
#
#     src_tokenizer = BertTokenizer.from_pretrained(src_lang_path)
#     trg_tokenizer = AutoTokenizer.from_pretrained(trg_lang_path)
#     new_tokenizer = BertTokenizer.from_pretrained(src_lang_path)
#
#     used_vocab = collections.OrderedDict()
#
#     default_vocab = itertools.islice(src_tokenizer.vocab.items(), 0, 106)
#     used_vocab.update(default_vocab)
#
#     src_vocab = src_tokenizer.get_vocab()
#     trg_vocab = trg_tokenizer.get_vocab()
#
#     for token, v in src_vocab.items():
#         if not token in used_vocab:
#             used_vocab.update({token: len(used_vocab)})
#     print(f"Copied used vocab to tokenizer fron source dictionary : {len(used_vocab)}")
#
#     for token, v in trg_vocab.items():
#         if not token in used_vocab:
#             used_vocab.update({token: len(used_vocab)})
#     print(f"Total Copied used vocab to tokenizer: {len(used_vocab)}")
#
#     new_token_path = os.path.join(args.root,f"{args.src}-{args.trg}-custom")
#     new_tokenizer.vocab = used_vocab
#     print(new_tokenizer)
#     new_tokenizer.save_pretrained(new_token_path)
#
#     for_validity= BertTokenizer.from_pretrained(new_token_path)
#     print(for_validity)
