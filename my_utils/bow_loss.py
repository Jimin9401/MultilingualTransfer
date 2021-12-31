import torch
from torch import nn
from dataclasses import dataclass

@dataclass
class BoWLoss:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.
    Args:
        epsilon (:obj:`float`, `optional`, defaults to 0.1):
            The label smoothing factor.
        ignore_index (:obj:`int`, `optional`, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100
    src_id: int = 0
    trg_id: int = 1

    def __call__(self, model_output, labels):
        # |model_output|: (batch_size, seq_len, vocab_size)
        # |labels|: (batch_size, seq_len)

        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        log_probs = -nn.functional.log_softmax(logits, dim=-1)

        seq_len = log_probs.size(1)

        bags = []
        special_token_ids=[0,1,2,-100, self.src_id, self.trg_id] # pad, sep
        
        for seq_label in labels:#.tolist():
            pad_start_idx = (1-seq_label.eq(self.ignore_index).long()).sum()
            if pad_start_idx < 4:
                bags.append(None)
            else:
                indices_to_gather = torch.zeros((pad_start_idx-2, pad_start_idx-3), dtype=torch.long)
                for i in range(1, pad_start_idx-2):
                    indices = seq_label[i+1:pad_start_idx-1]
                    dim = indices.size(-1)
                    indices_to_gather[i,:dim] = indices
                bags.append(indices_to_gather)
        #|bags[0]|: (1, B) -> (seq_len, B)

        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)
        # |labels| : (batch_size, seq_len, 1)

        padding_mask = labels.eq(self.ignore_index)

        
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels) # one-hot label
        
        bow_tgt_loss = 0.0
        num_active_elements = 0
        for i, bag in enumerate(bags):
            if bag is not None:
                bag = bag.to(labels.device)
                bow_loss = log_probs[i,:,:].gather(dim=-1, index=bag)
                bow_mask = bag.eq(0)
                bow_loss.masked_fill_(bow_mask, 0.0)
                num_active_elements += bow_mask.numel() - bow_mask.long().sum()
                bow_tgt_loss += bow_loss.sum()
            else:
                continue
        bow_tgt_loss = bow_tgt_loss / num_active_elements

        nll_loss.masked_fill_(padding_mask, 0.0)

        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements

        return (1 - self.epsilon) * nll_loss + self.epsilon * bow_tgt_loss


# @dataclass
# class BoWLoss:
#     """
#     Adds label-smoothing on a pre-computed output from a Transformers model.
#     Args:
#         epsilon (:obj:`float`, `optional`, defaults to 0.1):
#             The label smoothing factor.
#         ignore_index (:obj:`int`, `optional`, defaults to -100):
#             The index in the labels to ignore when computing the loss.
#     """

#     epsilon: float = 0.1
#     ignore_index: int = -100
#     src_id: int = 0
#     trg_id: int = 1

#     def __call__(self, model_output, labels):
#         # |model_output|: (batch_size, seq_len, vocab_size)
#         # |labels|: (batch_size, seq_len)

#         logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
#         log_probs = -nn.functional.log_softmax(logits, dim=-1)

#         seq_len = log_probs.size(1)

#         bags = []
#         special_token_ids=[0,1,2,-100, self.src_id, self.trg_id] # pad, sep
        
#         for seq_label in labels.tolist():
#             bow_index = list(set(seq_label)-set(special_token_ids))
#             bags.append(torch.LongTensor(bow_index).unsqueeze(0).repeat(seq_len, 1))     

#         #|bags[0]|: (1, B) -> (seq_len, B)

#         if labels.dim() == log_probs.dim() - 1:
#             labels = labels.unsqueeze(-1)
#         # |labels| : (batch_size, seq_len, 1)

#         padding_mask = labels.eq(self.ignore_index)

#         bow_mask = labels.eq(self.ignore_index)
#         for special_token_id in special_token_ids:
#             bow_mask += labels.eq(special_token_id)
#         # |bow_mask|: (batch_size, seq_len, 1)
#         # bow_mask[0,:,:].repeat(1,1,B)
        
#         # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
#         # will ignore them in any case.
#         labels = torch.clamp(labels, min=0)
#         # |log_probs| : (batch_size, seq_len, vocab_size)
#         nll_loss = log_probs.gather(dim=-1, index=labels) # one-hot label
#         # |nll_loss|: (batch_size, seq_len, 1)

#         # works for fp16 input tensor too, by internally upcasting it to fp32
#         # smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)
        
#         bow_tgt_loss = 0.0
#         num_active_elements = 0
#         for i, bag in enumerate(bags):
#             # |bag|: (1, B)
#             # |log_probs[i,:,:]|: (seq_len, vocab_size)
#             # print(f"BAG {i+1}")
#             # print(f"bag size:{bag.size()}")
#             bag = bag.to(labels.device)
#             bow_loss = log_probs[i,:,:].gather(dim=-1, index=bag)
#             # |bow_loss|: (seq_len, B)
#             B = bow_loss.size(1)
#             mask = bow_mask[i,:,:].repeat(1,B)
#             bow_loss.masked_fill_(mask, 0.0)
#             # print(f"mask_size: {mask.size()}")
#             # print(f"bow_loss size: {bow_loss.size()}")
#             num_active_elements += mask.numel() - mask.long().sum()
#             # print(f"mask.numel():{mask.numel()} | mask.long().sum(): {mask.long().sum()}")
#             # print("num_active_elements:", num_active_elements)
#             bow_tgt_loss += bow_loss.sum()
#             # print("bow_loss: ", bow_loss.size())
#             # print(bow_tgt_loss)
#             # |bow_mask[i,:,:]|: (seq_len, 1)
#             # |mask|: (seq_len, B)
#         bow_tgt_loss = bow_tgt_loss / num_active_elements

#         nll_loss.masked_fill_(padding_mask, 0.0)
#         # smoothed_loss.masked_fill_(padding_mask, 0.0)

#         # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
#         num_active_elements = padding_mask.numel() - padding_mask.long().sum()
#         nll_loss = nll_loss.sum() / num_active_elements

#         # smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
#         #return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss
#         return (1 - self.epsilon) * nll_loss + self.epsilon * bow_tgt_loss