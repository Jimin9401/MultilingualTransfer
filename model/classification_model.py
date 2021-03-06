from transformers import AutoModel, AutoConfig, BertModel, BertConfig, GPT2Model, AutoModelWithLMHead, BertForMaskedLM,GPT2LMHeadModel
import torch.nn as nn
import torch


class PretrainedTransformer(nn.Module):

    def __init__(self, args, encoder_class, n_class):
        super(PretrainedTransformer, self).__init__()
        self.args = args
        self.main_net = BertModel.from_pretrained(encoder_class)
        self.hidden_size = self.main_net.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, n_class)

    def forward(self, inps):
        out = self.main_net(input_ids=inps)
        hidden_states = out["last_hidden_state"]

        if self.args.prototype == "average":
            hidden = torch.mean(hidden_states, 1)
        elif self.args.prototype == "cls":
            hidden = hidden_states[:, 0]
        else:
            raise NotImplementedError

        return self.classifier(hidden_states[:, 0]), hidden

    def resize_token_embeddings(self, new_num_tokens):

        self.main_net.resize_token_embeddings(new_num_tokens)


