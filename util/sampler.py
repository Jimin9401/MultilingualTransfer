import torch
from .sampling_utils import *
from copy import deepcopy
from overrides import overrides

from model.language_model import GPT2Reassemble


class SamplerBase:
    def __init__(self, model, seq_length):
        self.model = model
        self.seq_length = seq_length

    def sample(self, inps, past):
        return NotImplementedError


class OneSampler(SamplerBase):
    def __init__(self, model: GPT2Reassemble, seq_length, top_whatever, stochastic=False, temperature: float = 1.0):
        """
        :param model:
        :param seq_length:
        :param stochastic: choice [top_k,top_p] if True
        """
        super(OneSampler, self).__init__(model, seq_length)

        self.temperature = temperature
        self.top_whatever = top_whatever
        if stochastic:
            self.sampling = top_sample
        else:
            self.sampling = greedy

    @torch.no_grad()
    def sample(self, inps):

        context = inps
        generated = deepcopy(inps)
        past = None

        for t in range(0, self.seq_length):
            lm_logits, past = self.model.forward_one(context, past_key_values=past)
            lm_logits = lm_logits / self.temperature
            lm_logits = lm_logits[:, -1]
            context = self.sampling(lm_logits, self.top_whatever, self.temperature)
            generated = torch.cat([generated, context], dim=-1)

        return generated


class BeamSampler(SamplerBase):
    def __init__(self, model, seq_length, beam_size: int = 3, temperature: float = 1.0):
        """
        no version on stochastic mode
        :param model:
        :param seq_length:
        :param top_whatever: int as beam_size
        """
        super(BeamSampler, self).__init__(model, seq_length)
        self.temperature = temperature
        # if not isinstance(beam_size, int):
        #     raise ValueError
        self.beam_size = beam_size
        self.sampling = greedy

    def _set_start_sequence(self, inps):
        batch, seq_lens = inps.size()
        res = inps[:, None].repeat(1, self.beam_size, 1)  # [batch, beam, l]
        res.view(-1, seq_lens)

        return res.view(-1, seq_lens)

    @torch.no_grad()
    def sample(self, inps):
        n_batch, seq_length = inps.size()
        context = self._set_start_sequence(inps)
        generated = deepcopy(context)
        past = None

        probs = torch.zeros([n_batch * self.beam_size]).to(context.device)
        for t in range(0, self.seq_length):
            lm_logits, past = self.model.forward_one(context, past=past)
            lm_logits = lm_logits[:, -1]

            context, probs, past, generated = self.beam_sample(lm_logits, probs, t, past, generated)

        return generated.cpu()[:, 0], probs

    def beam_sample(self, logits, probs, time_step, past, generated):

        if time_step == 0:
            logits = logits.view(-1, self.beam_size, logits.size()[-1])
            probs, preds = self.beam_start(logits, probs)
            generated = torch.cat([generated, preds], dim=-1)

        else:
            logits = logits.view(-1, self.beam_size, logits.size()[-1])
            probs, preds, past, generated = self.beam_continue(logits, probs, past, generated)

        return preds.view(-1, 1), probs, past, generated

    def beam_start(self, logits, probs):
        logits = logits / self.temperature
        p, i = torch.topk(torch.log_softmax(logits, -1), self.beam_size, -1)  # [batch, beam_size]
        i = i.view(-1, self.beam_size, self.beam_size)[:, 0, :].contiguous().view(-1, 1)
        p = p.view(-1, self.beam_size, self.beam_size)[:, 0, :].contiguous().view(-1, 1)

        probs = probs + p.view(-1)

        return probs, i

    def beam_continue(self, logits, probs, past, generated):
        bs = logits.size(0)
        generated = generated.view(bs, self.beam_size, -1)

        current_p, indexes = torch.topk(torch.log_softmax(logits, -1), self.beam_size,
                                        -1)  # [batch_size, beam_size, beam_size]
        probs = probs.view(bs, -1).unsqueeze(-1) + current_p
        new_probs = probs.view(bs, -1)

        probs, ni = new_probs.topk(self.beam_size, -1)
        sampled = indexes.view(bs, -1).gather(1, ni)  # [batch, beam]
        group = ni // self.beam_size
        ind = torch.arange(bs)[:, None], group
        generated = generated[ind]
        bs_beam = past[0][0].size(0)

        n_head, seq_len, hidden_size = past[0][0].size()[1:]

        past = [
            (k.view(bs, self.beam_size, n_head, seq_len, hidden_size)[ind].view(bs_beam, n_head, seq_len, hidden_size),
             v.view(bs, self.beam_size, n_head, seq_len, hidden_size)[ind].view(bs_beam, n_head, seq_len, hidden_size)) \
            for k, v in past]

        # sampled = indexes.view(bs, -1).gather(1, ni)
        generated = torch.cat([generated, sampled[:, :, None]], -1)

        return probs, sampled.view(-1)[:, None], past, generated
