from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader
import torch
import torch.nn as nn
import apex
from model.losses import NTXentLoss, AlignLoss
from model.classification_model import PretrainedTransformer
from overrides import overrides
from sklearn.metrics import classification_report, f1_score, accuracy_score
from itertools import chain
import math
import random


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


class LMTrainer(Trainer):
    def __init__(self, args, model, train_batchfier, test_batchfier, optimizers,
                 update_step, criteria, clip_norm, mixed_precision, n_label, scheduler):
        super(LMTrainer, self).__init__(args, model, train_batchfier, test_batchfier, optimizers,
                                        update_step, criteria, clip_norm, mixed_precision, scheduler)
        self.n_label = n_label

    def reformat_inp(self, inp):
        inp_tensor = tuple(i.to(self.args.gpu) for i in inp)
        return inp_tensor

    def train_epoch(self):

        model = self.model
        batchfier = self.train_batchfier
        criteria = self.criteria
        optimizer = self.optimizers

        if isinstance(batchfier, IterableDataset):
            print("asdads")
            if self.args.distributed_training:
                print("distributed")
                from .parallel import DynamicDistributedDataLoader
                batchfier = DynamicDistributedDataLoader(dataset=batchfier,
                                                         batch_size=batchfier.size,
                                                         shuffle=False,
                                                         collate_fn=batchfier.collate, pin_memory=True)
            else:
                batchfier = DataLoader(dataset=batchfier,
                                       batch_size=batchfier.size,
                                       shuffle=False,
                                       collate_fn=batchfier.collate, pin_memory=True)

        model.train()
        tot_loss, step_loss, tot_cnt, n_bar, acc = 0, 0, 0, 0, 0
        pbar_cnt = 0
        model.zero_grad()
        pbar = tqdm(batchfier, total=batchfier.dataset.num_buckets)

        for inp in pbar:
            inp, attn_mask, gt = self.reformat_inp(inp)
            logits = model(input_ids=inp, attention_mask=attn_mask)
            if random.random() > self.args.teacher_forcing_ratio:
                gt = torch.argmax(logits, -1)
            loss = criteria(logits.view(-1, logits.size(-1)), gt.view(-1))

            step_loss += loss.item()
            tot_loss += loss.item()
            tot_cnt += 1

            if self.mixed_precision:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if not tot_cnt % self.update_step:
                self.step += 1
                pbar_cnt += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
                optimizer.step()
                self.scheduler.step(self.step)
                model.zero_grad()
                pbar.set_description(
                    "training loss : {0:4f}, ppl : {1:4f}, lr : {2:5f}".format(
                        step_loss / (self.update_step * pbar_cnt), math.exp(step_loss / (self.update_step * pbar_cnt)),
                        self.scheduler.get_last_lr()[0]), )
                pbar.update()
                # if pbar_cnt == 100:
                #     pbar, n_bar, pbar_cnt, step_loss, acc = reset_pbar(pbar, n_bar)

        pbar.close()

    def test_epoch(self):
        model = self.model
        batchfier = self.test_batchfier

        if self.args.dataset == "bio_ner":
            batchfier.collate = batchfier.collate_ner

        if isinstance(self.criteria, tuple):
            _, criteria = self.criteria
        else:
            criteria = self.criteria

        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate, pin_memory=True)

        model.eval()
        model.zero_grad()
        pbar = tqdm(batchfier)
        pbar_cnt = 0
        step_loss = 0
        tot_score = 0.0

        true_buff = []
        eval_buff = []

        for inp in pbar:
            with torch.no_grad():
                inp, attn_mask, gt = self.reformat_inp(inp)
                logits = model(input_ids=inp, attention_mask=attn_mask)
                loss = criteria(logits.view(-1, logits.size(-1)), gt.view(-1))
                preds = torch.argmax(logits, -1)
                preds = preds.view(-1)
                step_loss += loss.item()
                pbar_cnt += 1

            true_buff.append(gt.view(-1).tolist())
            eval_buff.append(preds.view(-1).tolist())
            score = torch.mean((preds.view(-1) == gt.view(-1)).to(torch.float))
            tot_score += score

            pbar.set_description(
                "test loss : %f, perplexity : %f, accuracy : %f" % (
                    step_loss / pbar_cnt, math.exp(step_loss / pbar_cnt), tot_score / pbar_cnt), )
            pbar.update()
        pbar.close()
        true_buff = list(chain(*true_buff))
        eval_buff = list(chain(*eval_buff))
        accuracy = accuracy_score(true_buff, eval_buff)

        if self.args.dataset == "chemprot":
            f1 = f1_score(true_buff, eval_buff, labels=list(range(0, self.n_label)), average="micro")
        else:
            f1 = f1_score(true_buff, eval_buff, labels=list(range(0, self.n_label)), average="macro")

        print("test accuracy: {0:.4f}  test f1: {1:.4f}".format(accuracy, f1))

        return accuracy, math.exp(step_loss / pbar_cnt)


from model.nmt_model import CustomMBart


class NMTTrainer(Trainer):
    def __init__(self, args, model: CustomMBart, train_batchfier, test_batchfier, optimizers,
                 update_step, criteria, clip_norm, mixed_precision, scheduler):
        super(NMTTrainer, self).__init__(args, model, train_batchfier, test_batchfier, optimizers,
                                         update_step, criteria, clip_norm, mixed_precision, scheduler)

    def reformat_inp(self, inp):
        inp_tensor = tuple(i.to("cuda") for i in inp)
        return inp_tensor

    def train_epoch(self):
        model = self.model
        batchfier = self.train_batchfier
        optimizer = self.optimizers

        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate)

        model.train()
        tot_loss, step_loss, tot_cnt, n_bar, acc = 0, 0, 0, 0, 0

        pbar_cnt = 0
        model.zero_grad()
        pbar = tqdm(batchfier, total=batchfier.dataset.num_buckets)

        for inputs in pbar:
            # inputs = self.reformat_inp(inp)
            labels = inputs["labels"]
            outputs = model(**inputs)  #
            logits = outputs["logits"]

            loss = self.criteria(logits.view(-1, logits.size(-1)), labels.view(-1))
            step_loss += loss.item()
            tot_cnt += 1

            if self.mixed_precision:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if not tot_cnt % self.update_step:
                self.step += 1
                pbar_cnt += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
                optimizer.step()
                self.scheduler.step(self.step)
                model.zero_grad()
                ppl = math.exp(step_loss / (self.update_step * pbar_cnt))
                pbar.set_description(
                    "training loss : %f, ppl: %f , iter : %d" % (
                        step_loss / (self.update_step * pbar_cnt), ppl,
                        n_bar), )
                pbar.update()
            if pbar_cnt % 50==0:
                wandb.log({
                    "Train perplexity": math.exp(loss.item()),
                    "Train Loss": loss.item()})


        pbar.close()

    def test_epoch(self):
        model = self.model
        batchfier = self.test_batchfier

        if isinstance(self.criteria, tuple):
            _, criteria = self.criteria
        else:
            criteria = self.criteria

        if self.args.mbart:
            test_criteria = nn.CrossEntropyLoss(ignore_index=1)
        else:
            test_criteria = nn.CrossEntropyLoss(ignore_index=0)

        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate)

        model.eval()
        # cached_data_loader=get_cached_data_loader(batchfier,batchfier.size,custom_collate=batchfier.collate,shuffle=False)
        model.zero_grad()
        pbar = tqdm(batchfier)
        pbar_cnt = 0
        step_loss = 0

        for inputs in pbar:
            with torch.no_grad():
                labels = inputs["labels"]
                outputs = model(**inputs)  #
                logits = outputs["logits"]

                loss = test_criteria(logits.view(-1, logits.size(-1)), labels.view(-1))
                step_loss += loss.item()
                pbar_cnt += 1

            pbar.set_description(
                "test loss : %f  test perplexity : %f" % (
                    step_loss / pbar_cnt, math.exp(step_loss / pbar_cnt)), )
            pbar.update()
        pbar.close()

        wandb.log({
            "Test perplexity": math.exp(step_loss / pbar_cnt),
            "Test Loss": step_loss / pbar_cnt})

        return step_loss / pbar_cnt,  math.exp(step_loss / pbar_cnt)



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