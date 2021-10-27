from .sampler import BeamSampler, OneSampler
from model.language_model import GPT2Reassemble
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch


class EvaluatorBase:
    def __init__(self, args, model: GPT2Reassemble):
        self.args = args
        self.model = model
        self.step = 0

    def reformat_inp(self, inp):
        raise NotImplementedError

    def generate_epoch(self, dataset):
        raise NotImplementedError


class LMEvaluator(EvaluatorBase):
    def __init__(self, args, model: GPT2Reassemble, nprefix=50, temperature=1.0):
        super(LMEvaluator, self).__init__(args, model)
        self.nprefix = nprefix
        if self.args.beam:
            print(args.top_whatever)
            self.sampler = BeamSampler(model, seq_length=self.args.ngenerate, beam_size=args.top_whatever,
                                       temperature=temperature)

        else:
            is_stochastic = args.decoding_strategy == "stochastic"
            self.sampler = OneSampler(model, seq_length=self.args.ngenerate, top_whatever=args.top_whatever,
                                      stochastic=is_stochastic, temperature=temperature)

    def reformat_inp(self, inp):
        inp_tensor = tuple(i.to(self.args.gpu) for i in inp)
        return inp_tensor

    def generate_epoch(self, batchfier):

        def truncate(x, prefix_len):
            return [i[prefix_len:] for i in x]

        prefixs = []
        truths = []
        generated = []

        batchfier = DataLoader(dataset=batchfier,
                               batch_size=batchfier.size,
                               shuffle=False,
                               num_workers=1,
                               collate_fn=batchfier.collate, pin_memory=True)

        batchfier = [batch for batch in batchfier]

        for inp in tqdm(batchfier, total=len(batchfier)):
            inp, attn_mask, gt = self.reformat_inp(inp)
            prefix = inp[:, :self.nprefix]
            gt = gt[:, self.nprefix:self.args.ngenerate + self.nprefix]
            # if gt.size(-1) < self.args.ngenerate:
            #     continue
            predict, _ = self.sampler.sample(prefix)
            generated.extend(predict.cpu()[:, self.nprefix:].tolist())
            truths.extend(gt.cpu().tolist())
            prefixs.extend(prefix.cpu().tolist())

        return pd.DataFrame({'prefix': prefixs, 'decoded_predict': generated, 'decoded_true': truths})


from model.nmt_model import CustomMBart


class NMTEvaluator(EvaluatorBase):
    def __init__(self, args, model: CustomMBart, tokenizer, trg_id: int):
        super(NMTEvaluator, self).__init__(args, model)
        self.tokenizer = tokenizer
        self.trg_id = trg_id
        self.model.eval()

    def generate_epoch(self, batchfier):
        source = []
        truths = []
        generated = []

        batchfier = DataLoader(dataset=batchfier,
                               batch_size=batchfier.size,
                               shuffle=False,
                               collate_fn=batchfier.collate)

        batchfier = [batch for batch in batchfier]

        pbar = tqdm(batchfier)

        for inputs in pbar:
            input_ids = inputs["input_ids"]
            attn_mask = inputs["attention_mask"]
            gt = inputs["labels"]

            with torch.no_grad():
                # encoder_output = self.model.model.encoder(input_ids=input_ids, attention_mask=attn_mask)
                if self.args.beam:
                    outputs = self.model.generate(input_ids=input_ids,
                                              forced_bos_token_id=self.trg_id, num_beams=self.args.top_k)  #
                else:
                    outputs = self.model.generate(input_ids=input_ids,
                                                  forced_bos_token_id=self.trg_id)

            generated.extend(outputs.cpu().tolist())
            truths.extend(gt.cpu().tolist())
            source.extend(input_ids.cpu().tolist())

        return pd.DataFrame({'source': source, 'decoded_predict': generated, 'decoded_true': truths})
