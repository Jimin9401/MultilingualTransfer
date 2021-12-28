from contextlib import contextmanager
import os
from typing import Any, Dict, List, Optional, Tuple

from tokenizers import SentencePieceBPETokenizer

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging


logger = logging.get_logger(__name__)


class NMTTokenizer(PreTrainedTokenizer):

    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

    def __init__(
        self,
        vocab_filename,
        merges_filename,
        src_lang=None,
        tgt_lang=None,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=None,
        **kwargs
    ):
        super().__init__(
            bos_token = bos_token,
            eos_token = eos_token,
            sep_token = sep_token,
            cls_token = cls_token,
            unk_token = unk_token,
            pad_token = pad_token,
        )
        
        self.bpe_model = SentencePieceBPETokenizer.from_file(
            vocab_filename,
            merges_filename,
        )

        self.vocab = self.bpe_model.get_vocab()
        self.id2token = {v: k for (k, v) in self.vocab.items()}

        self._src_lang = src_lang if src_lang is not None else "en_XX"
        self.cur_lang_code_id = self.vocab[self._src_lang]
        self.tgt_lang = tgt_lang
        self.set_src_lang_special_tokens(self._src_lang)
    
    def get_vocab(self):
        return self.vocab

    @property
    def vocab_size(self):
        return len(self.vocab)
    
    @property
    def src_lang(self) -> str:
        return self._src_lang
    
    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)

    def _tokenize(self, text: str) -> List[str]:
        return self.bpe_model.encode(text).tokens

    def _convert_token_to_id(self, token):
        return self.vocab[token]
    
    def _convert_id_to_token(self, index):
        return self.id2token[index]

    
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:

        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens
    
    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this bpe_model")
        self.src_lang = src_lang
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
        inputs["forced_bos_token_ids"] = tgt_lang_id
        return inputs
    
    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_lang: str = "en_XX",
        tgt_texts: Optional[List[str]]=None,
        tgt_lang: str = "ko_KR",
        **kwargs,
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    @contextmanager
    def as_target_tokenizer(self):
        self.set_tgt_lang_special_tokens(self.tgt_lang)
        yield
        self.set_src_lang_special_tokens(self.src_lang)
    
    def set_src_lang_special_tokens(self, src_lang) -> None:
        self.cur_lang_code = self.vocab[src_lang]
        self.prefix_tokens = [self.cur_lang_code]
        self.suffix_tokens = [self.eos_token_id]
    
    def set_tgt_lang_special_tokens(self, tgt_lang) -> None:
        self.cur_lang_code = self.vocab[tgt_lang]
        self.prefix_tokens = [self.cur_lang_code]
        self.suffix_tokens = [self.eos_token_id]

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace("▁", " ").strip()
        return out_string

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        spaces_between_special_tokens: bool = True,
        **kwargs
    ) -> str:

        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separately for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            elif token in ['vi_VN', 'en_XX', 'ko_KR', 'tr_TR', 'th_TH', 'mk_MK', 'he_IL', 'hr_HR', 'uk_UA', 'fa_IR']:
                continue
            text.append(token)
        if text:
            text = self.convert_tokens_to_string(text)
        else:
            text = " "
        return text