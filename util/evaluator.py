from .sampler import BeamSampler, OneSampler
from model.language_model import GPT2Reassemble
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

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

        for inp in tqdm(batchfier,total=len(batchfier)):
            inp, attn_mask, gt = self.reformat_inp(inp)
            prefix = inp[:, :self.nprefix]
            gt = gt[:, self.nprefix:self.args.ngenerate+self.nprefix]
            # if gt.size(-1) < self.args.ngenerate:
            #     continue
            predict, _ = self.sampler.sample(prefix)
            generated.extend(predict.cpu()[:,self.nprefix:].tolist())
            truths.extend(gt.cpu().tolist())
            prefixs.extend(prefix.cpu().tolist())

        return pd.DataFrame({'prefix': prefixs, 'decoded_predict': generated, 'decoded_true': truths})



class NMTEvaluator(EvaluatorBase):
    def __init__(self, args, model: GPT2Reassemble, nprefix=50, temperature=1.0):
        super(NMTEvaluator, self).__init__(args, model)
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

        for inp in tqdm(batchfier,total=len(batchfier)):
            inp, attn_mask, gt = self.reformat_inp(inp)
            prefix = inp[:, :self.nprefix]
            gt = gt[:, self.nprefix:self.args.ngenerate+self.nprefix]
            # if gt.size(-1) < self.args.ngenerate:
            #     continue
            predict, _ = self.sampler.sample(prefix)
            generated.extend(predict.cpu()[:,self.nprefix:].tolist())
            truths.extend(gt.cpu().tolist())
            prefixs.extend(prefix.cpu().tolist())

        return pd.DataFrame({'prefix': prefixs, 'decoded_predict': generated, 'decoded_true': truths})
