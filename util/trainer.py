from re import L
import os
from copy import deepcopy
from tqdm import tqdm
from model.nmt_model import CustomMBart
from torch.utils.data import IterableDataset, DataLoader
import torch
from transformers import DataCollatorForSeq2Seq
#import apex
from model.losses import NTXentLoss, AlignLoss
from model.classification_model import PretrainedTransformer
from overrides import overrides
from sklearn.metrics import classification_report, f1_score, accuracy_score
from itertools import chain
import math
import random
from datasets import load_metric


class Collate4Regularization:
    def __init__(self, args, tokenizer, teacher_tokenizer, label_pad_token_id=-100):
        self.args = args
        self.tokenizer = tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.label_pad_token_id = label_pad_token_id
    
    def __call__(self, features):
        import numpy as np

        main_inputs = []
        auxiliary_inputs = []

        labels = [feature[0]["labels"] for feature in features] if "labels" in features[0][0].keys() else None
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature[0]["labels"]))
                if isinstance(feature[0]["labels"], list):
                    feature[0]["labels"] = (
                        feature[0]["labels"] + remainder if padding_side == "right" else remainder + feature[0]["labels"]
                    )
                elif padding_side == "right":
                    feature[0]["labels"] = np.concatenate([feature[0]["labels"], remainder]).astype(np.int64)
                else:
                    feature[0]["labels"] = np.concatenate([remainder, feature[0]["labels"]]).astype(np.int64)
        
        for (main_feature, auxiliary_feature) in features:
            main_inputs.append(main_feature)
            auxiliary_inputs.append(auxiliary_feature)

        main_inputs = self.tokenizer.pad(
            main_inputs,
            padding=True,
            max_length=self.args.seq_len,
            return_tensors='pt'
        )

        auxiliary_inputs = self.teacher_tokenizer.pad(
            auxiliary_inputs,
            padding=True,
            max_length=self.args.seq_len,
            return_tensors='pt'
        )
        return main_inputs, auxiliary_inputs

def collate_fn(data):
    inputs = {}
    keys = ['src_input_ids', 'tgt_input_ids', 'src_attention_mask', 'tgt_attention_mask',\
            'gt_src_input_ids', 'gt_tgt_input_ids', 'gt_src_attention_mask', 'gt_tgt_attention_mask',\
            'n_sub_tokens_per_src_group', 'n_sub_tokens_per_tgt_group', 'gt_n_sub_tokens_per_src_group', 'gt_n_sub_tokens_per_tgt_group']
    for data_item in data:
        for key in keys:
            inputs[key]=torch.LongTensor([data_item[key]])
    return inputs


class Trainer:
    def __init__(self, args, model: PretrainedTransformer, train_batchfier, test_batchfier, optimizers,
                 update_step, criteria, clip_norm, mixed_precision, scheduler):
        self.args = args
        self.model = model
        self.train_batchfier = train_batchfier
        self.test_batchfier = test_batchfier
        self.optimizers = optimizers
        self.criteria = criteria
        self.step = 0
        self.update_step = update_step
        self.mixed_precision = mixed_precision
        self.clip_norm = clip_norm
        self.scheduler = scheduler

    def reformat_inp(self, inp):
        raise NotImplementedError

    def train_epoch(self):
        return NotImplementedError

    def test_epoch(self):
        return NotImplementedError


class NMTTrainer(Trainer):
    def __init__(self, args, model: CustomMBart, train_batchfier, test_batchfier, optimizers,
                 update_step, criteria, clip_norm, mixed_precision, scheduler, tokenizer, **kwargs):
        super(NMTTrainer, self).__init__(args, model, train_batchfier, test_batchfier, optimizers,
                                         update_step, criteria, clip_norm, mixed_precision, scheduler)
        
        if isinstance(tokenizer, dict):
            self.trg_id = tokenizer['tgt'].lang_code
            self.decoder_start_token_id = tokenizer['tgt'].decoder_start_token_id
            print("decoder_start_token_id: ", self.decoder_start_token_id)
            self.src_tokenizer = tokenizer['src']
            self.tgt_tokenizer = tokenizer['tgt']
        else:
            self.trg_id = tokenizer.convert_tokens_to_ids(tokenizer.tgt_lang)        
            self.src_tokenizer = tokenizer
            self.tgt_tokenizer = tokenizer
        
        if 'teacher_encoder' in kwargs:
            self.teacher_encoder = kwargs['teacher_encoder']
        else:
            self.teacher_encoder = None

        if 'auxiliary_criteria' in kwargs:
            self.auxiliary_criteria = kwargs['auxiliary_criteria']
        else:
            self.auxiliary_criteria = None

        if 'pretrained_tokenizer' in kwargs:
            self.pretrained_tokenizer = kwargs['pretrained_tokenizer']
        else:
            self.pretrained_tokenizer = None

        if 'zeroing_optimizer' in kwargs:
            self.zeroing_optimizer = kwargs['zeroing_optimizer']
            print('zeroing optimizer is set!')
        else:
            self.zeroing_optimizer = None

    def reformat_inp(self, inp):
        for key in inp.keys():
            inp[key] = inp[key].to("cuda")
        return inp

    def train_epoch(self):
        model = self.model
        batchfier = self.train_batchfier
        optimizer = self.optimizers
        teacher_encoder = self.teacher_encoder 
        
        if self.mixed_precision:
            scaler = torch.cuda.amp.GradScaler(enabled=True)

        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate)
            pbar = tqdm(batchfier, total=batchfier.dataset.num_buckets)

        else:
            if self.args.replace_vocab_w_existing_lm_tokenizers:
                label_pad_token_id = self.tgt_tokenizer.padding_id #if args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
            else:
                label_pad_token_id = -100
            if self.args.train_with_regularization:
                collate_fn = Collate4Regularization(self.args, tokenizer=self.src_tokenizer, teacher_tokenizer=self.pretrained_tokenizer)
            else:
                collate_fn = DataCollatorForSeq2Seq(tokenizer=self.src_tokenizer, label_pad_token_id=label_pad_token_id)

            batchfier = DataLoader(
                dataset = batchfier,
                batch_size = self.args.per_gpu_train_batch_size,
                shuffle=True,
                collate_fn=collate_fn
                )
            
            pbar = tqdm(batchfier, total=len(batchfier))

        model.train()
        tot_loss, step_loss, tot_cnt, n_bar, acc = 0, 0, 0, 0, 0

        pbar_cnt = 0
        model.zero_grad()

        print("current learning rate:", self.scheduler.get_last_lr()[0])


        for inputs in pbar:
            if self.args.train_with_regularization:
                inputs, aux_inputs = inputs
                aux_inputs = self.reformat_inp(aux_inputs)
            else:
                aux_inputs = None
            inputs = self.reformat_inp(inputs)
            if tot_cnt == 0:
                print("LABELS\n")
                print(inputs['labels'][0])
                example = deepcopy(inputs['labels'][0])
                temp_label_mask = example.eq(-100)
                example = example.masked_fill_(temp_label_mask, self.tgt_tokenizer.pad_token_id)
                print(self.tgt_tokenizer.convert_ids_to_tokens(example))
                print("================================\n")
                del example
                del temp_label_mask

                print("INPUTS:\n")
                print(self.src_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
            labels = inputs["labels"]
            if self.mixed_precision:
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = model(**inputs)  #
                    logits = outputs["logits"]
                    loss = self.criteria(logits.view(-1, logits.size(-1)), labels.view(-1))
                    step_loss += loss.item()
                
                    loss = loss / self.update_step    
                scaler.scale(loss).backward()

            else:
                if aux_inputs is not None:
                    outputs = model(**inputs, output_hidden_states=True)
                    mask = inputs["attention_mask"]
                    denominator = mask.sum(dim=1).unsqueeze(-1)
                    student_output = outputs.encoder_last_hidden_state * mask.unsqueeze(-1)
                    student_output = student_output.sum(dim=1).unsqueeze(1)
                    student_output = student_output / denominator

                    with torch.no_grad():
                        teacher_output = teacher_encoder(**aux_inputs)
                        #|teacher_output|: (batch_size, seq_len, hidden_size)
                        #|mask_size|: (batch_size, seq_len)
                        mask = aux_inputs["attention_mask"]
                        denominator = mask.sum(dim=1).unsqueeze(-1)
                        teacher_output = teacher_output.last_hidden_state * mask.unsqueeze(-1)
                        teacher_output = teacher_output.sum(dim=1).unsqueeze(1)
                        teacher_output = teacher_output / denominator

                else:
                    outputs = model(**inputs)
               
                if self.args.label_smoothing_factor != 0:
                    loss = self.criteria(outputs, labels)
                else:
                    logits = outputs["logits"]
                    loss = self.criteria(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                this_step_loss = loss.item()
                
                ## auxiliary loss (calculated from sentence-level regularization)
                if aux_inputs is not None:
                    auxiliary_loss = self.auxiliary_criteria(student_output, teacher_output)            
                    this_step_auxiliary_loss = auxiliary_loss.item()
                if aux_inputs is not None:
                    step_loss += (this_step_loss + this_step_auxiliary_loss)
                    final_loss = loss + auxiliary_loss
                    final_loss.backward()
                else:
                    step_loss += this_step_loss
                    this_step_auxiliary_loss=0.0
                    loss.backward()

            tot_cnt += 1

            # if self.mixed_precision:
                # with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()

            if not tot_cnt % self.update_step:
                self.step += 1
                pbar_cnt += 1
                if self.mixed_precision:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
                    # print(scaler.get_scale())
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    if self.args.initial_freeze:
                        torch.nn.utils.clip_grad_norm_(model.model.shared.parameters(), self.clip_norm)
                    else:                
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)                
                    optimizer.step()

                    

                # self.scheduler.step() #TBC??
                optimizer.zero_grad()
                if self.zeroing_optimizer is not None:
                    self.zeroing_optimizer.zero_grad()
                # model.zero_grad()

                # ppl = math.exp(step_loss / (self.update_step * pbar_cnt))
                # ppl = math.exp(step_loss / (self.update_step * pbar_cnt))
                ppl = math.exp(this_step_loss)
                
                pbar.set_description(
                    "training loss : %f, auxiliary loss: %f,  ppl: %f , iter : %d" % (
                        this_step_loss, this_step_auxiliary_loss, ppl,
                        n_bar), )
                # pbar.set_description(
                #     "training loss : %f, ppl: %f , iter : %d" % (
                #         step_loss / (self.update_step * pbar_cnt), ppl,
                #         n_bar), )
                pbar.update(self.update_step)
            if pbar_cnt % 50==0:
                wandb.log({
                    "Train perplexity": math.exp(loss.item()),
                    "Train Loss": loss.item()})
            
            self.scheduler.step()

        # self.scheduler.step()
        pbar.close()

    def test_epoch(self, epoch=None):
        model = self.model
        batchfier = self.test_batchfier

        # if isinstance(self.criteria, tuple):
        #     _, criteria = self.criteria
        # else:
        #     criteria = self.criteria

        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate)
        
        else:
            if self.args.train_with_regularization:
                eval_collate_fn = Collate4Regularization(self.args, tokenizer=self.src_tokenizer, teacher_tokenizer=self.pretrained_tokenizer, label_pad_token_id=self.tgt_tokenizer.pad_token_id)
            else:
                eval_collate_fn = DataCollatorForSeq2Seq(tokenizer=self.src_tokenizer, label_pad_token_id=self.tgt_tokenizer.pad_token_id)

            batchfier = DataLoader(
                dataset = batchfier,
                batch_size = self.args.per_gpu_eval_batch_size,
                shuffle=False,
                collate_fn=eval_collate_fn
                )
            pbar = tqdm(batchfier, total=len(batchfier))

        model.eval()
        # cached_data_loader=get_cached_data_loader(batchfier,batchfier.size,custom_collate=batchfier.collate,shuffle=False)
        model.zero_grad()
        pbar = tqdm(batchfier)
        pbar_cnt = 0
        step_loss = 0#DELETE

        metric = load_metric("sacrebleu")

        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [[label.strip()] for label in labels]
            return preds, labels

        # file2write_path = ' '
        # os.makedirs(file2write_path, exist_ok=True)
        # replace_vocab = "replace_vocab" if self.args.replace_vocab else ""
        # file2write_name = os.path.join(file2write_path, f"{self.args.src}-{self.args.trg}-{replace_vocab}-{self.args.vocab_size}-epoch_{epoch}-results.txt")
        # file2write2_name = os.path.join(file2write_path, f"{self.args.src}-{self.args.trg}-{replace_vocab}-{self.args.vocab_size}-epoch_{epoch}-output.txt")

        # file2write = open(file2write_name, 'w')
        # file2write2 = open(file2write2_name, 'w')


        for inputs in pbar:

            if self.args.train_with_regularization:
                inputs, aux_inputs = inputs

            inputs = self.reformat_inp(inputs)

            input_ids = inputs["input_ids"]
            
            gt = inputs["labels"]

            with torch.no_grad():
                if self.args.beam:
                    if self.args.replace_vocab_w_existing_lm_tokenizers:
                        outputs = self.model.generate(
                            input_ids=input_ids,
                            # decoder_start_token_id=self.trg_id,
                            decoder_start_token_id=self.decoder_start_token_id,
                            num_beams=self.args.top_k,
                            length_penalty=self.args.length_penalty,
                            eos_token_id=self.tgt_tokenizer.eos_token_id,
                        )
                    else:
                        outputs = self.model.generate(
                            input_ids=input_ids,
                            forced_bos_token_id=self.trg_id,
                            num_beams=self.args.top_k,
                            length_penalty=self.args.length_penalty,
                            eos_token_id=self.tgt_tokenizer.eos_token_id,
                        )
                else:
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        forced_bos_token_id=self.trg_id,
                    )

                out = self.tgt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                ref = self.tgt_tokenizer.batch_decode(gt, skip_special_tokens=True)

            CNT=0
            print(outputs)
            for out_sent, ref_sent in zip(out, ref):
                # file2write.write(f"Example {CNT+1}\n")
                print(f"Example {CNT+1}\n")
                
                # file2write.write(f"model output: {out_sent}\n")
                print(f"model output: {out_sent}\n")
                # file2write2.write(f"{out_sent}\n")
                # file2write.write(f"GT: {ref_sent}\n")
                print(f"GT: {ref_sent}\n")
                # file2write.write("==================================\n")
                print("==================================\n")
                CNT +=1

            out, ref = postprocess_text(out, ref)

            metric.add_batch(predictions=out, references=ref)

            pbar_cnt += 1

            pbar.update()
        pbar.close()
        # file2write.close()
        # file2write2.close()

        eval_metric = metric.compute()
        print(f"epoch: {epoch}\nevaluation bleu: {eval_metric['score']}\n")

        wandb.log({
            "Eval blue": eval_metric["score"],
            })

        return eval_metric



import wandb


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    example_images = []

    test_loss /= len(test_loader.dataset)


    wandb.log({
        "Examples": example_images,
        "Test PPL": 100. * correct / len(test_loader.dataset),
        "Test Loss": test_loss})