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

## NEW VERSION ! 
def align_vocabularies(source_tokenizer: MBart50Tokenizer, target_tokenizer: SentencePieceBPETokenizer):
    source_vocab = source_tokenizer.get_vocab()
    target_vocab = target_tokenizer.get_vocab()

    new_dict = {}
    remainder = []
    for word, idx in target_vocab.items():
        try:
            if word in source_vocab:
                original_ids = source_tokenizer.convert_tokens_to_ids([word])
            else:      
            # original_ids = [source_vocab[i] for i in source_tokenizer.tokenize(word)]
                original_ids = source_tokenizer.tokenize(word)
                original_ids = source_tokenizer.convert_tokens_to_ids(original_ids)
        except:
            original_ids = []
            remainder.append(word)

        # if len(original_ids) > 1 and original_ids[0] == 6:
        #     original_ids = original_ids[1:]
        new_dict[idx] = original_ids

    print(len(remainder))
    # for special tokens
    # for i in range(0, 3):
    #     new_dict[i] = [i]
    # new_dict[0] = [3]
    return new_dict

def align_vocabularies_w_bert_tokenizer(source_tokenizer: MBart50Tokenizer, target_tokenizer, language='en'):
    from util.data_builder import LMAP
    source_vocab = source_tokenizer.get_vocab()
    target_vocab = target_tokenizer.get_vocab()

    new_dict = {}
    remainder = []

    special_tokens_map = {
        '[CLS]': '<s>',
        '[PAD]': '<pad>',
        '[SEP]': '</s>',
        '[UNK]': '<unk>',
        '[MASK]': '<mask>',
    }

    for word, idx in target_vocab.items():
        try:
            if word in special_tokens_map:
                new_dict[idx] = source_tokenizer.convert_tokens_to_ids([special_tokens_map[word]])
                continue

            if word == '[unused1]':
                new_dict[idx] = source_tokenizer.convert_tokens_to_ids([LMAP[language]])
                target_tokenizer.lang_code = idx
                target_tokenizer.language = language
                continue

            if word.startswith('##'):
                word = word[2:]
            else:
                word = "▁" + word

            if word in source_vocab:
                original_ids = source_tokenizer.convert_tokens_to_ids([word])
            else:
                original_ids = source_tokenizer.tokenize(word)
                original_ids = source_tokenizer.convert_tokens_to_ids(original_ids)
        except:
            original_ids = []
            remainder.append(word)

        # if len(original_ids) > 1 and original_ids[0] == 6:
        #     original_ids = original_ids[1:]
        new_dict[idx] = original_ids

    print(len(remainder))
    # for special tokens
    # for i in range(0, 3):
    #     new_dict[i] = [i]
    # new_dict[0] = [3]
    return new_dict

def align_vocabularies_w_bart_tokenizer(source_tokenizer: MBart50Tokenizer, target_tokenizer, language='ko'):
    from util.data_builder import LMAP
    source_vocab = source_tokenizer.get_vocab()
    target_vocab = target_tokenizer.get_vocab()

    new_dict = {}
    for word, idx in target_vocab.items():
        if word == '<usr>':
            original_ids = source_tokenizer.convert_tokens_to_ids([LMAP[language]])
            target_tokenizer.language = language
            target_tokenizer.lang_code = idx
            print("TARGET_TOKENIZER_LANG_CODE: ", idx)
        elif word in source_vocab:
            original_ids = source_tokenizer.convert_tokens_to_ids([word])
            if word == '</s>':
                target_tokenizer.decoder_start_token_id = idx
                print("TARGET_TOKENIZER_DECODER_START_TOKEN_ID:", idx)
            elif word == '<pad>':
                target_tokenizer.padding_id = idx
                print("TARGET_TOKENIZER_PAD_ID:", idx)
            elif idx == 0:
                print(word, idx, "***")
        else:      
            original_ids = source_tokenizer.tokenize(word)
            original_ids = source_tokenizer.convert_tokens_to_ids(original_ids)

        new_dict[idx] = original_ids
        
    return new_dict

# def align_vocabularies_w_bart_tokenizer(source_tokenizer: MBart50Tokenizer, target_tokenizer, language='ko'):
#     from util.data_builder import LMAP
#     source_vocab = source_tokenizer.get_vocab()
#     target_vocab = target_tokenizer.get_vocab()

#     new_dict = {}
#     remainder = []
#     for word, idx in target_vocab.items():
#         try:
#             if word in source_vocab:
#                 original_ids = source_tokenizer.convert_tokens_to_ids([word])
#             else:      
#                 original_ids = source_tokenizer.tokenize(word)
#                 original_ids = source_tokenizer.convert_tokens_to_ids(original_ids)
#         except:
#             original_ids = []
#             remainder.append(word)

#         new_dict[idx] = original_ids
    
#     # add language code
#     print(idx+1)
#     new_dict[idx+1] = source_tokenizer.convert_tokens_to_ids([LMAP[language]])
#     target_tokenizer.lang_code = idx+1
#     target_tokenizer.language = language
    
#     print(len(remainder))
#     return new_dict

# PREVIOUS VERSION
# def align_vocabularies(source_tokenizer: MBart50Tokenizer, target_tokenizer: SentencePieceBPETokenizer):
#     source_vocab = source_tokenizer.get_vocab()
#     target_vocab = target_tokenizer.get_vocab()

#     new_dict = {}
#     remainder = []
#     for word, idx in target_vocab.items():
#         try:
#             # original_ids = [source_vocab[i] for i in source_tokenizer.tokenize(word)]
#             original_ids = source_tokenizer.tokenize(word)
#             original_ids = source_tokenizer.convert_tokens_to_ids(original_ids)
#         except:
#             original_ids = []
#             remainder.append(word)

#         if len(original_ids) > 1 and original_ids[0] == 6:
#             original_ids = original_ids[1:]
#         new_dict[idx] = original_ids

#     print(len(remainder))
#     # for special tokens
#     # for i in range(0, 3):
#     #     new_dict[i] = [i]
#     # new_dict[0] = [3]

#     return new_dict

def align_vocabularies_with_dictionary(source_tokenizer, target_tokenizer, dic_file):#, keep_polysemy=True):
    source_vocab = source_tokenizer.get_vocab()
    target_vocab = target_tokenizer.get_vocab()

    with open(dic_file, 'r') as f:
        dic = f.read().splitlines()
    
    bilingual_dictionary = {}

    for alignment in dic:
        src_word, english_word = alignment.split("\t")
        if src_word in bilingual_dictionary:
            bilingual_dictionary[src_word].append(english_word)
        else:
            bilingual_dictionary[src_word] = [english_word]
    
    filtered_dictionary = {}
    for word in bilingual_dictionary:
        ids = target_tokenizer.tokenize(word)
        if len(ids) > 1 and ids[0] ==  "▁":
            ids = ids[1:]
        if len(ids) == 1:
            filtered_dictionary[word] = bilingual_dictionary[word]
    
    del bilingual_dictionary
    print(f"Use {len(filtered_dictionary)} bilingual alignments !!!")

    new_dict = {}
    remainder = []
    cnt = 0
    for word, idx in target_vocab.items():
        try:
            original_ids = source_tokenizer.tokenize(word)
            # filtering process
            if len(original_ids) > 1 and original_ids[0] == "▁":
                original_ids = original_ids[1:]

            if len(original_ids) > 1: #('아예 같은 경우 제외')
                if word in filtered_dictionary:
                    cnt += 1
                    print(word)
                    english_words = filtered_dictionary[word]
                    if len(english_words)==1:
                        original_ids = source_tokenizer.tokenize(english_words)
                        if len(original_ids) > 1 and original_ids[0] == "▁":
                            original_ids = original_ids[1:]
                    else:
                        original_ids = []
                        for english_word in english_words:
                            ids = source_tokenizer.tokenize(english_word)
                            if len(ids) > 1 and ids[0] == "▁":
                                ids = ids[1:]
                            original_ids.extend(ids)

            original_ids = source_tokenizer.convert_tokens_to_ids(original_ids)
        except:
            original_ids = []
            remainder.append(word)

        if len(original_ids) > 1 and original_ids[0] == 6:
            original_ids = original_ids[1:]
        new_dict[idx] = original_ids

    print(len(remainder))
    print(remainder)
    print(cnt)
    # for special tokens
    # for i in range(0, 3):
    #     new_dict[i] = [i]
    # new_dict[0] = [3]

    return new_dict


