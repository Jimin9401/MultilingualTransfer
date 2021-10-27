from util.trainer import NMTTrainer
from util.data_builder import load_dataset
from util.args import NMTArgument
from tqdm import tqdm
import pandas as pd
import wandb
# from embedding_utils.embedding_initializer import transfer_embedding
# from corpus_utils.bpe_mapper import CustomTokenizer
from transformers import AdamW, MBart50Tokenizer, MBartConfig
# from t import WarmupLinearSchedule
import apex
from util.batch_generator import NMTBatchfier
# from model.classification_model import PretrainedTransformer
from model.nmt_model import CustomMBart
import torch.multiprocessing as mp

from transformers import get_scheduler

import torch.nn as nn
import torch
import random
from util.logger import *
import logging

logger = logging.getLogger(__name__)


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_trainer(args, model, train_batchfier, test_batchfier, tokenizer):
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

    # optimizer=RAdam(model.parameters(),args.learning_rate,weight_decay=args.weight_decay)
    # optimizer = AdamW(model.model.shared.parameters(), args.lr, weight_decay=args.weight_decay)

    if args.initial_freeze:
        optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.model.shared.parameters(), args.lr, weight_decay=args.weight_decay)

    if args.mixed_precision:
        print('mixed_precision')
        opt_level = 'O2'
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level=opt_level)

        # from apex.parallel import DistributedDataParallel as DDP
        # model=DDP(model,delay_allreduce=True)

    if torch.cuda.device_count() > 1:

        if args.distributed_training:
            from util.parallel import set_init_group
            model = set_init_group(model, args)

        else:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
    # decay_step = args.decay_step
    # decay_step=0
    # scheduler = WarmupLinearSchedule(optimizer, args.warmup_step, args.decay_step)

    criteria = nn.CrossEntropyLoss(ignore_index=1)

    if args.max_train_steps is None:
        args.num_update_steps_per_epoch = train_batchfier.num_buckets
        args.max_train_steps = args.n_epoch * args.num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    trainer = NMTTrainer(args, model, train_batchfier, test_batchfier, optimizer, args.gradient_accumulation_step,
                         criteria, args.clip_norm, args.mixed_precision, lr_scheduler)

    return trainer


def get_batchfier(args, tokenizer: MBart50Tokenizer):
    n_gpu = torch.cuda.device_count()
    train, dev, test = load_dataset(args, tokenizer, "nmt")
    padding_idx = 1

    train_batch = NMTBatchfier(args, train, batch_size=args.per_gpu_train_batch_size * n_gpu, maxlen=args.seq_len,
                               padding_index=padding_idx, device="cuda")
    dev_batch = NMTBatchfier(args, dev, batch_size=args.per_gpu_eval_batch_size * n_gpu, maxlen=args.seq_len,
                             padding_index=padding_idx, device="cuda")
    test_batch = NMTBatchfier(args, test, batch_size=args.per_gpu_eval_batch_size * n_gpu, maxlen=args.seq_len,
                              padding_index=padding_idx, device="cuda")

    return train_batch, dev_batch, test_batch


def run(gpu, args):
    # args = ExperimentArgument()
    args.gpu = gpu
    args.device = gpu
    args.aug_ratio = 0.0
    set_seed(args.seed)

    print(args.__dict__)
    pretrained_config = MBartConfig.from_pretrained(args.encoder_class)

    from util.data_builder import LMAP
    model = CustomMBart.from_pretrained(args.encoder_class)

    if args.replace_vocab:
        from tokenizers import SentencePieceBPETokenizer
        vocab_json_name = args.vocab_path + f"/{args.src}-{args.trg}-vocab.json"
        merge_name = args.vocab_path + f"/{args.src}-{args.trg}-merges.txt"

        print(vocab_json_name)
        print(merge_name)

        deployed_tokenizer = SentencePieceBPETokenizer.from_file(vocab_json_name, merge_name)
        pretrained_tokenizer = MBart50Tokenizer.from_pretrained(args.encoder_class, src_lang=LMAP[args.src],
                                                                tgt_lang=LMAP[args.trg])
        from corpus_utils.vocab_util import align_vocabularies

        new_dict = align_vocabularies(pretrained_tokenizer, deployed_tokenizer)
        special_map = {k: v for k, v in
                       zip(pretrained_tokenizer.additional_special_tokens,
                           pretrained_tokenizer.additional_special_tokens_ids)}
        special_ids = [special_map[LMAP[args.src]], special_map[LMAP[args.trg]]]
        args.new_special_src_id = len(new_dict)
        args.new_special_trg_id = len(new_dict) + 1
        model.rearrange_token_embedding(new_dict, special_ids)

    else:
        special_ids = [LMAP[args.src], LMAP[args.trg]]
        deployed_tokenizer = MBart50Tokenizer.from_pretrained(MBARTCLASS, src_lang=LMAP[args.src],
                                                              tgt_lang=LMAP[args.trg])

    args.extended_vocab_size = 0
    train_gen, dev_gen, test_gen = get_batchfier(args, deployed_tokenizer)

    model.to("cuda")
    wandb.watch(model)
    trainer = get_trainer(args, model, train_gen, dev_gen, deployed_tokenizer)
    best_dir = os.path.join(args.savename, "best_model")

    if not os.path.isdir(best_dir):
        os.makedirs(best_dir)

    results = []

    optimal_perplexity = 1000.0
    not_improved = 0

    if args.do_train:
        for e in tqdm(range(0, args.n_epoch)):
            print("Epoch : {0}".format(e))

            # if args.initial_freeze and e < args.initial_epoch_for_rearrange:
            #     for params in model.model.encoder.parameters():
            #         params.requires_grad=False
            #     for params in model.model.decoder.parameters():
            #         params.requires_grad=False
            #     trainer.train_epoch()
            # else:
            #     for params in model.parameters():
            #         params.requires_grad=True
            trainer.train_epoch()

            save_path = os.path.join(args.savename, "epoch_{0}".format(e))
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            if args.evaluate_during_training:
                loss, step_perplexity = trainer.test_epoch()
                results.append({"eval_loss": loss, "eval_ppl": step_perplexity})

                if optimal_perplexity > step_perplexity:
                    optimal_perplexity = step_perplexity
                    # if args.distributed_training:
                    #     torch.save(model.module.state_dict(), os.path.join(best_dir, "best_model.bin"))
                    # else:
                    torch.save(model.state_dict(), os.path.join(best_dir, "best_model.bin"))
                    print("Update Model checkpoints at {0}!! ".format(best_dir))
                    not_improved = 0
                else:
                    not_improved += 1

            if not_improved >= 5:
                break

        log_full_eval_test_results_to_file(args, config=pretrained_config, results=results)

    if args.do_eval:
        accuracy, macro_f1 = trainer.test_epoch()
        descriptions = os.path.join(args.savename, "eval_results.txt")
        writer = open(descriptions, "w")
        writer.write("accuracy: {0:.4f}, macro f1 : {1:.4f}".format(accuracy, macro_f1) + "\n")
        writer.close()

    if args.do_test:
        # original_tokenizer = MBart50Tokenizer.from_pretrained(args.encoder_class)
        # args.aug_word_length = len(tokenizer) - len(original_tokenizer)
        # trainer.test_batchfier = test_gen
        from util.evaluator import NMTEvaluator
        model = CustomMBart.from_pretrained(args.encoder_class)
        if args.replace_vocab:
            model.resize_token_embeddings(len(new_dict) + 2)
            model.final_logits_bias.data = torch.zeros([1, 250054])

        model.to(args.gpu)

        state_dict = torch.load(os.path.join(args.checkpoint_name_for_test, "best_model", "best_model.bin"))

        model.load_state_dict(state_dict)
        if args.replace_vocab:
            shape = model.final_logits_bias.data.shape
            model.final_logits_bias.data = torch.zeros(shape)  # we remove final logit bias when training

        if args.replace_vocab:
            trg_id = len(new_dict) + 1
        else:

            idx=deployed_tokenizer.additional_special_tokens.index(LMAP[args.trg])
            special_ids = deployed_tokenizer.additional_special_tokens_ids
            trg_id =special_ids[idx]

        evaluator = NMTEvaluator(args, model, tokenizer=deployed_tokenizer, trg_id=trg_id)

        if not os.path.isdir(args.test_file):
            os.makedirs(args.test_file)

        output = evaluator.generate_epoch(test_gen)
        if args.beam:
            pd.to_pickle(output, os.path.join(args.test_file, "result-beam.pkl"))
        else:
            pd.to_pickle(output, os.path.join(args.test_file, "result-greedy.pkl"))

        # log_full_test_results_to_file(args, config=pretrained_config)


if __name__ == "__main__":

    args = NMTArgument()
    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.ngpus_per_node
    project_name=f"{args.src}-{args.trg}"

    if args.replace_vocab:
        project_name+="-replace_vocab"

    if args.initial_freeze:
        project_name += "-embedding_only"


    wandb.init(project=project_name, reinit=True)
    wandb.config.update(args)

    if args.distributed_training:
        if args.ngpus_per_node < 2:
            raise ValueError("Require ngpu>=2")

        mp.spawn(run, nprocs=args.ngpus_per_node, args=(args,))
    else:
        run("cuda", args)
