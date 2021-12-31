from transformers import MBartForConditionalGeneration, MBartConfig
from transformers.models.mbart import MBartForConditionalGeneration
#from fairseq.models.transformer import transformer_wmt_en_de

# from transformers.modeling_bart import BartForConditionalGeneration

import torch
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss
import math

from transformers.modeling_outputs import (
    Seq2SeqLMOutput,
)

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
    """
    Shift input ids one token to the right, and wrap the last non pad token (the <LID> token) Note that MBart does not
    have a single `decoder_start_token_id` in contrast to other Bart-like models.
    """
    prev_output_tokens = input_ids.clone()

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)

    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens


class CustomMBart(MBartForConditionalGeneration):

    def __init__(self, config: MBartConfig):
        super().__init__(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id) # config.pad_token_id
                #decoder_input_ids = shift_tokens_right(labels, 3) # config.pad_token_id

        # print(decoder_input_ids)
        # print(decoder_input_ids[0].tolist())


        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) # + self.final_logits_bias

        masked_lm_loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()
        #     masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        #
        # if not return_dict:
        #     output = (lm_logits,) + outputs[1:]
        #     return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


    def rearrange_token_embedding(self, new_dict):#, special_ids: list):
        # new_weight = torch.randn([len(new_dict) + 2, self.config.hidden_size])
        new_weight = torch.randn([len(new_dict), self.config.hidden_size])
        # new_bias = torch.randn([1, len(new_dict) + 2])

        pretrained_word_embedding = self.model.encoder.embed_tokens.weight.data

        # src_id, trg_id = special_ids
        # src_map_id, trg_map_id = len(new_dict), len(new_dict) + 1

        for new_idx, original_idx in new_dict.items():
            if len(original_idx)>0:
                original_idx = torch.LongTensor(original_idx)
                new_weight[new_idx] = torch.mean(pretrained_word_embedding[original_idx], 0)

        # new_weight[src_map_id] = pretrained_word_embedding[src_id]
        # new_weight[trg_map_id] = pretrained_word_embedding[trg_id]

        self.model.encoder.embed_tokens.weight.data = new_weight
        self.lm_head.weight.data = new_weight # shared embedding

    def rearrange_token_embedding_w_existing_lm_tokenizers(self, new_dict):
        # new_weight = torch.randn([len(new_dict) + 2, self.config.hidden_size])
        src_dict = new_dict['src']
        tgt_dict = new_dict['tgt']
        
        new_encoder_emb_weight = torch.randn([len(src_dict), self.config.hidden_size])
        new_decoder_emb_weight = torch.randn([len(tgt_dict), self.config.hidden_size])
        # print(new_encoder_emb_weight.size())
        # print(new_decoder_emb_weight.size())
        # new_bias = torch.randn([1, len(new_dict) + 2])

        pretrained_word_embedding = self.model.encoder.embed_tokens.weight.data

        # src_id, trg_id = special_ids
        # src_map_id, trg_map_id = len(new_dict), len(new_dict) + 1

        for new_idx, original_idx in src_dict.items():
            if len(original_idx)>0:
                original_idx = torch.LongTensor(original_idx)
                new_encoder_emb_weight[new_idx] = torch.mean(pretrained_word_embedding[original_idx], 0)

        for new_idx, original_idx in tgt_dict.items():
            if len(original_idx)>0:
                original_idx = torch.LongTensor(original_idx)
                new_decoder_emb_weight[new_idx] = torch.mean(pretrained_word_embedding[original_idx], 0)

        self.model.encoder.embed_tokens.weight.data = new_encoder_emb_weight
        self.model.decoder.embed_tokens.weight.data = new_decoder_emb_weight
        self.lm_head.weight = self.model.decoder.embed_tokens.weight


class Adapter(torch.nn.Module):
    def __init__(self, embedding_layer, pos_embedding_layer, max_seq_len=512):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.pos_embedding_layer = pos_embedding_layer
        self.embed_scale = math.sqrt(self.embedding_layer.embedding_dim)
        self.max_seq_len = max_seq_len

    def forward(
        self,
        input_ids
    ):

        input_shape = input_ids.size()
        if input_shape[-1] > 512:
            return None
        
        else:
            input_ids = input_ids.view(-1, input_shape[-1])

            inputs_embeds = self.embedding_layer(input_ids) * self.embed_scale

            embed_pos = self.pos_embedding_layer(input_shape)
            hidden_states = inputs_embeds + embed_pos
        return hidden_states
