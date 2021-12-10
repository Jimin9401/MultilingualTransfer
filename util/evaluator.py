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
                                              decoder_start_token_id=self.trg_id, num_beams=self.args.top_k)  #
                else:
                    outputs = self.model.generate(input_ids=input_ids,
                                                  decoder_start_token_id=self.trg_id,eos_token_id=self.tokenizer.sep_token_id)

            generated.extend(outputs.cpu().tolist())
            truths.extend(gt.cpu().tolist())
            source.extend(input_ids.cpu().tolist())

        return pd.DataFrame({'source': source, 'decoded_predict': generated, 'decoded_true': truths})
