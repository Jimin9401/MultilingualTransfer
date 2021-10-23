from util.trainer import LMTrainer
from util.data_builder import get_dataset
from util.args import ExperimentArgument
from tqdm import tqdm
import pandas as pd
from embedding_utils.embedding_initializer import transfer_embedding
from corpus_utils.bpe_mapper import CustomTokenizer
from transformers import BertTokenizer, AdamW, BertConfig
# from pytorch_transformers import WarmupLinearSchedule
import apex
from util.batch_generator import
# from model.classification_model import PretrainedTransformer
from model.language_model import GPT2Reassemble, GPT2Config, GPT2Tokenizer,GPT2Generation

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
    # optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)

    # optimizer=RAdam(model.parameters(),args.learning_rate,weight_decay=args.weight_decay)
    optimizer = AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

    if args.mixed_precision:
        print('mixed_precision')
        opt_level = 'O2'
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level=opt_level)
        # from apex.parallel import DistributedDataParallel as DDP
        # model=DDP(model,delay_allreduce=True)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        # model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu])

    # decay_step = args.decay_step
    # decay_step=0
    # scheduler = WarmupLinearSchedule(optimizer, args.warmup_step, args.decay_step)
    criteria = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab_size)

    if args.max_train_steps is None:
        args.num_update_steps_per_epoch = train_batchfier.num_buckets
        args.max_train_steps = args.n_epoch * args.num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    trainer = LMTrainer(args, model, train_batchfier, test_batchfier, optimizer, lr_scheduler,
                        args.gradient_accumulation_step, criteria, args.clip_norm, args.mixed_precision,
                        args.n_label)

    return trainer


def get_batchfier(args, tokenizer):
    n_gpu = torch.cuda.device_count()
    train, dev, test, label = get_dataset(args, tokenizer)

    if isinstance(tokenizer, tuple):
        _, domain_tokenizer = tokenizer
        padding_idx = domain_tokenizer.pad_token_id
        mask_idx = domain_tokenizer.pad_token_id
    else:
        padding_idx = tokenizer.vocab_size
        mask_idx = tokenizer.vocab_size
        # padding_idx = 0
        # mask_idx = 0
        print(tokenizer.vocab_size)

    # if args.contrastive:
    #     train_batch = ContrastiveBatchFier(args, train, batch_size=args.per_gpu_train_batch_size * n_gpu,
    #                                        maxlen=args.seq_len,
    #                                        padding_index=padding_idx, mask_idx=mask_idx)
    #     dev_batch = ContrastiveBatchFier(args, dev, batch_size=args.per_gpu_eval_batch_size * n_gpu,
    #                                      maxlen=args.seq_len,
    #                                      padding_index=padding_idx, mask_idx=mask_idx)
    #     test_batch = ContrastiveBatchFier(args, test, batch_size=args.per_gpu_eval_batch_size * n_gpu,
    #                                       maxlen=args.seq_len,
    #                                       padding_index=padding_idx, mask_idx=mask_idx)

    # else:
    train_batch = LMBatchFier(args, train, batch_size=args.per_gpu_train_batch_size * n_gpu, maxlen=args.seq_len,
                              padding_index=padding_idx)
    dev_batch = LMBatchFier(args, dev, batch_size=args.per_gpu_eval_batch_size * n_gpu, maxlen=args.seq_len,
                            padding_index=padding_idx)
    test_batch = LMBatchFier(args, test, batch_size=args.per_gpu_eval_batch_size * n_gpu, maxlen=args.seq_len,
                             padding_index=padding_idx)

    # from torch.utils.data import DataLoader
    # train_batchfier=DataLoader(train_batch,batch_size=train_batch.size,collate_fn=train_batch.collate)
    # dev_batchfier=DataLoader(dev_batch, batch_size=dev_batch.size*4, collate_fn=dev_batch.collate,)

    return train_batch, dev_batch, test_batch, label


def run(gpu,args):
    # args = ExperimentArgument()
    args.gpu=gpu
    args.aug_ratio = 0.0
    set_seed(args.seed)
    args.gpu = gpu

    print(args.__dict__)
    pretrained_config = GPT2Config.from_pretrained(args.encoder_class, pad_token="<PAD>")

    if args.init_embed:
        from tokenizers import ByteLevelBPETokenizer
        encoder_class = ByteLevelBPETokenizer
        print(args.vocab_path)
        tokenizer = CustomTokenizer(args=args, dir_path=args.root, encoder_class=encoder_class,
                                    dataset_name=args.dataset, vocab_size=args.vocab_size)
        print(tokenizer.encoder)
        args.vocab_size = tokenizer.encoder.get_vocab_size()
        tokenizer_for_transfer = GPT2Tokenizer.from_pretrained(args.encoder_class)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(args.encoder_class)
        args.extended_vocab_size = 0

    # logger.info("\nNew merged Vocabulary size is %s" % (args.extended_vocab_size))

    train_gen, dev_gen, test_gen, label = get_batchfier(args, tokenizer)
    args.n_label = len(label)

    inverse_label_map = {v: k for k, v in label.items()}
    args.label_list = inverse_label_map

    model = GPT2Generation.from_pretrained(args.encoder_class)
    # model = GPT2Reassemble(pretrained_config,args.encoder_class)

    if args.init_embed:
        pretrained_vocab = tokenizer_for_transfer.get_vocab()
        custom_vocab = tokenizer.encoder.get_vocab()
        intersection = {}
        remains = []

        for cvk in custom_vocab.keys():
            if cvk in pretrained_vocab:
                intersection[custom_vocab[cvk]] = pretrained_vocab[cvk]
            else:
                remains.append((cvk,custom_vocab[cvk]))
        remain_vocab = {}
        indicator = 'Ġ'

        for remain in remains:

            key, value = remain
            if indicator in key:
                key = " " + key[1:]
            a = tokenizer_for_transfer.encode(key)
            remain_vocab[value] = a

        model.init_embeddings(intersection,remain_vocab)


    # if args.merge_version:
    #     d2p = pd.read_pickle(os.path.join(args.vocab_path, "d2p.pickle"))
    #     expand_token_embeddings(model, tokenizer)
    #     embedding(args, model, d2p)
    model.transformer.resize_token_embeddings(tokenizer.vocab_size + 1)

    model.cuda(args.gpu)

    trainer = get_trainer(args, model, train_gen, dev_gen, tokenizer)
    best_dir = os.path.join(args.savename, "best_model")

    if not os.path.isdir(best_dir):
        os.makedirs(best_dir)

    results = []

    optimal_perplexity = 1000.0
    not_improved = 0

    if args.do_train:
        for e in tqdm(range(0, args.n_epoch)):
            print("Epoch : {0}".format(e))
            trainer.train_epoch()
            save_path = os.path.join(args.savename, "epoch_{0}".format(e))
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            if args.evaluate_during_training:
                accuracy, step_perplexity = trainer.test_epoch()
                results.append({"eval_acc": accuracy, "eval_ppl": step_perplexity})

                if optimal_perplexity > step_perplexity:
                    optimal_perplexity = step_perplexity
                    torch.save(model.state_dict(), os.path.join(best_dir, "best_model.bin"))
                    print("Update Model checkpoints at {0}!! ".format(best_dir))
                    not_improved = 0
                else:
                    not_improved += 1

            if not_improved >= 10:
                break

        log_full_eval_test_results_to_file(args, config=pretrained_config, results=results)

    if args.do_eval:
        accuracy, macro_f1 = trainer.test_epoch()
        descriptions = os.path.join(args.savename, "eval_results.txt")
        writer = open(descriptions, "w")
        writer.write("accuracy: {0:.4f}, macro f1 : {1:.4f}".format(accuracy, macro_f1) + "\n")
        writer.close()

    if args.do_test:
        original_tokenizer = GPT2Tokenizer.from_pretrained(args.encoder_class)
        args.aug_word_length = len(tokenizer) - len(original_tokenizer)
        trainer.test_batchfier = test_gen
        from util.evaluator import LMEvaluator


        if args.model_path_list == "":
            raise EnvironmentError("require to clarify the argment of model_path")

        model_path = [model_path for model_path in args.model_path_list][0]

        state_dict = torch.load(os.path.join(model_path, "best_model", "best_model.bin"))
        model.load_state_dict(state_dict)

        results = []
        evaluator = LMEvaluator(args, model, args.nprefix,args.temperature)

        if args.model_path_list == "":
            raise EnvironmentError("require to clarify the argment of model_path")

        output = evaluator.generate_epoch(test_gen)
        pd.to_pickle(output,args.test_file)

        log_full_test_results_to_file(args, config=pretrained_config, results=results)


if __name__ == "__main__":

    args = ExperimentArgument()
    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.ngpus_per_node

    if args.distributed_training:
        if args.ngpus_per_node==1:
            raise ValueError("Require ngpu>=2")
        import torch.multiprocessing as mp
        mp.spawn(run, nprocs=args.ngpus_per_node, args=(args,))
    else:
        run("cuda", args)
