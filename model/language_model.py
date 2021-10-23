from transformers import GPT2Model, GPT2Config, GPT2PreTrainedModel, GPT2Tokenizer, GPT2LMHeadModel
import torch.nn as nn
from copy import deepcopy
# from transformers.modeling_gpt2 import GPT2Model
import torch

class GPT2Reassemble(GPT2PreTrainedModel):
    def __init__(self, config, encoder_class):
        super().__init__(config)
        self.config = config
        self.transformer = GPT2LMHeadModel.from_pretrained(encoder_class)
        self.transformer.tie_weights()

    def init_embeddings(self, intersection: dict, remains: dict):
        embedding_matrix = nn.Parameter(torch.rand(self.config.vocab_size, self.config.n_embd))
        nn.init.xavier_uniform_(embedding_matrix)

        for key, value in intersection.items():
            embedding_matrix.data[key] = self.transformer.transformer.wte.weight.data[value]

        for key, values in remains.items():
            embedding_matrix.data[key] = torch.mean(self.transformer.transformer.wte.weight.data[values], dim=0)

        print(f"Preserve Vocabulary size : {len(intersection)}")
        print(f"Initialized Vocabulary size : {len(remains)}")

        self.transformer.transformer.wte.weight = deepcopy(embedding_matrix)
        self.transformer.tie_weights()

        # self.transformer.set_input_embeddings(embedding_matrix)

    def forward(
            self,
            input_ids=None,
            attention_mask=None):
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask)
        # hidden_states = transformer_outputs[0]

        # lm_logits = self.lm_head(hidden_states)
        # outputs = (lm_logits,) + transformer_outputs[1:]

        return outputs[0]  # (loss), lm_logits, presents, (all hidden_states), (attentions)

    def forward_one(self, input_ids=None, attention_mask=None, past=None):
        outputs = self.transformer.transformer(input_ids=input_ids, past_key_values=past, attention_mask=attention_mask)

        hidden_states, presents = outputs[0:2]
        lm_logits = self.transformer.lm_head(hidden_states)

        # hidden_states = transformer_outputs[0]

        # lm_logits = self.lm_head(hidden_states)
        # outputs = (lm_logits,) + transformer_outputs[1:]

        return lm_logits, presents  # (loss), lm_logits, presents, (all hidden_states), (attentions)


# from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel, T5ForConditionalGeneration
# import torch.nn as nn
# import torch
# from overrides import overrides


class GPT2Generation(GPT2LMHeadModel):

    def __init__(self, config):
        super().__init__(config)

    def init_embeddings(self, intersection: dict, remains: dict):
        embedding_matrix = nn.Parameter(torch.rand(self.config.vocab_size, self.config.n_embd))
        nn.init.xavier_uniform_(embedding_matrix)

        for key, value in intersection.items():
            embedding_matrix.data[key] = self.transformer.wte.weight.data[value]

        for key, values in remains.items():
            embedding_matrix.data[key] = torch.mean(self.transformer.wte.weight.data[values], dim=0)

        print(f"Preserve Vocabulary size : {len(intersection)}")
        print(f"Initialized Vocabulary size : {len(remains)}")

        self.transformer.wte.weight = deepcopy(embedding_matrix)
        self.tie_weights()


    def forward(self, input_ids=None, past_key_values=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, labels=None, use_cache=None, output_attentions=None,
                output_hidden_states=None):
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        return lm_logits


    def forward_one(self, input_ids=None, past=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, labels=None, use_cache=None, output_attentions=None,
                output_hidden_states=None):

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states, presents = transformer_outputs[0:2]
        lm_logits = self.lm_head(hidden_states)
        # outputs = (lm_logits,) + transformer_outputs[1:]

        return lm_logits, presents  # (loss), lm_logits, presents, (all hidden_states), (attentions)
