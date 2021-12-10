from util.trainer import NMTTrainer
from util.data_builder import load_dataset, get_ted_dataset, get_dataset
from util.args import NMTArgument, InitialArgument
from tqdm import tqdm
import pandas as pd
import wandb
from transformers import AdamW, MBart50Tokenizer, MBartConfig, get_scheduler,AutoTokenizer, BertTokenizer
from util.batch_generator import NMTBatchfier
from model.nmt_model import CustomMBart
import torch.multiprocessing as mp
import apex
import torch.nn as nn
import torch
import random
from util.logger import *
import logging

logger = logging.getLogger(__name__)

LANGDICT = {"ko": "monologg/koelectra-base-discriminator",
            "es": 'dccuchile/bert-base-spanish-wwm-cased',
            "fi": "TurkuNLP/bert-base-finnish-uncased-v1",
            "ja": "cl-tohoku/bert-base-japanese",
            "tr": "dbmdz/bert-base-turkish-cased",
            "id": "indolem/indobert-base-uncased"}


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_trainer(args, model, train_batchfier, test_batchfier, tokenizer):
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    if args.mixed_precision:
        print('mixed_precision')
        opt_level = 'O2'
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level=opt_level)

    if torch.cuda.device_count() > 1:

        if args.distributed_training:
            from util.parallel import set_init_group
            model = set_init_group(model, args)

        else:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

    from model.losses import LabelSmoothingLoss

    # criteria = nn.CrossEntropyLoss(ignore_index=1,label_smoothing=0.3
    if args.mbart:
        criteria = nn.CrossEntropyLoss(ignore_index=1, label_smoothing=0.1)
    else:
        criteria = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)

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

def create_shared_vocabulary(args,src_tokenizer,tgt_tokenizer):
    custom_vocab_path = os.path.join(args.root, "vocab", f"{args.src}-{args.trg}")

    src_vocab = src_tokenizer.get_vocab()
    tgt_vocab = tgt_tokenizer.get_vocab()
    tgt_tokenizer.save_pretrained(custom_vocab_path)


    f = open(os.path.join(custom_vocab_path, "vocab.txt"), "a")
    for key in src_vocab:
        if key not in tgt_vocab:
            f.write(key + "\n")
    f.close()

    tokenizer = BertTokenizer.from_pretrained(custom_vocab_path)

    return tokenizer

def get_batchfier(args, tokenizer: MBart50Tokenizer, class_of_dataset):
    n_gpu = torch.cuda.device_count()
    train, dev, test = get_ted_dataset(args, tokenizer, class_of_dataset)
    # train, dev, test = get_dataset(args, tokenizer)

    if args.mbart:
        padding_idx = 1
    else:
        padding_idx = tokenizer.pad_token_id

    train_batch = NMTBatchfier(args, train, batch_size=args.per_gpu_train_batch_size * n_gpu, maxlen=args.seq_len,
                               padding_index=padding_idx, device="cuda")
    dev_batch = NMTBatchfier(args, dev, batch_size=args.per_gpu_eval_batch_size * n_gpu, maxlen=args.seq_len,
                             padding_index=padding_idx, device="cuda")
    test_batch = NMTBatchfier(args, test, batch_size=args.per_gpu_eval_batch_size * n_gpu, maxlen=args.seq_len,
                              padding_index=padding_idx, device="cuda",shuffle=False)

    return train_batch, dev_batch, test_batch


def run(gpu, args):
    # args = ExperimentArgument()
    args.gpu = gpu
    args.device = gpu
    args.aug_ratio = 0.0
    set_seed(args.seed)
    print(args.__dict__)

    from util.data_builder import LMAP

    if args.mbart:
        tokenizer = MBart50Tokenizer.from_pretrained(args.encoder_class, src_lang=LMAP[args.src],
                                                     tgt_lang=LMAP[args.trg])
        from util.dataset import ParallelDataset
        class_of_dataset = ParallelDataset
        vocab_size = len(tokenizer)

    else:
        src_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        # src_vocab = src_tokenizer.get_vocab()
        vocab_path = LANGDICT[args.trg]

        tgt_tokenizer = AutoTokenizer.from_pretrained(vocab_path)
        # custom_vocab_path = os.path.join(args.root, "vocab", f"{args.src}-{args.trg}")
        # tgt_tokenizer.save_pretrained(custom_vocab_path)

        # tgt_vocab = tgt_tokenizer.get_vocab()

        # f = open(os.path.join(custom_vocab_path, "vocab.txt"), "a")
        # for key in src_vocab:
        #     if key not in tgt_vocab:
        #         f.write(key + "\n")
        # f.close()

        tokenizer = create_shared_vocabulary(args,src_tokenizer,tgt_tokenizer)

        # tokenizer = BertTokenizer.from_pretrained(custom_vocab_path)


        # from corpus_utils.bpe_mapper import CustomNMTTokenizer
        from util.dataset import TEDParallelDataset
        # tokenizer = CustomNMTTokenizer(src_tokenizer, tgt_tokenizer)
        class_of_dataset = TEDParallelDataset
        vocab_size = len(tokenizer)

    from transformers import MBartForConditionalGeneration, MBartConfig
    mbart_config = MBartConfig.from_pretrained("facebook/mbart-large-50")

    mbart_config.encoder_layers = 5
    mbart_config.decoder_layers = 5
    mbart_config.d_model = 512
    mbart_config.encoder_attention_heads = 8
    mbart_config.decoder_attention_heads = 8
    mbart_config.decoder_ffn_dim = 2048
    mbart_config.decoder_ffn_dim = 2048


    model = CustomMBart(config=mbart_config)

    if not args.mbart:
        # model.model.shared = nn.Embedding(vocab_size + 2, model.config.d_model, padding_idx=0)
        model.resize_token_embeddings(vocab_size + 2)
        model.init_weights()

    print(model)
    model.to("cuda")
    wandb.watch(model)

    train_gen, dev_gen, test_gen = get_batchfier(args, tokenizer, class_of_dataset)
    trainer = get_trainer(args, model, train_gen, dev_gen, tokenizer)
    best_dir = os.path.join(args.savename, "best_model")

    if not os.path.isdir(best_dir):
        os.makedirs(best_dir)

    results = []

    optimal_perplexity = 100000.0
    not_improved = 0

    if args.do_train:
        for e in tqdm(range(0, args.n_epoch)):
            print("Epoch : {0}".format(e))
            trainer.train_epoch()

            save_path = os.path.join(args.savename, "epoch_{0}".format(e))
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            if args.evaluate_during_training:
                loss, step_perplexity = trainer.test_epoch()
                results.append({"eval_loss": loss, "eval_ppl": step_perplexity})

                if optimal_perplexity > step_perplexity:
                    optimal_perplexity = step_perplexity

                    torch.save(model.state_dict(), os.path.join(best_dir, "best_model.bin"))
                    print("Update Model checkpoints at {0}!! ".format(best_dir))
                    not_improved = 0
                else:
                    not_improved += 1

            if not_improved >= 5:
                break

        log_full_eval_test_results_to_file(args, config=mbart_config, results=results)

    if args.do_eval:
        accuracy, macro_f1 = trainer.test_epoch()
        descriptions = os.path.join(args.savename, "eval_results.txt")
        writer = open(descriptions, "w")
        writer.write("accuracy: {0:.4f}, macro f1 : {1:.4f}".format(accuracy, macro_f1) + "\n")
        writer.close()

    if args.do_test:
        from util.evaluator import NMTEvaluator

        mbart_config.encoder_layers = 5
        mbart_config.decoder_layers = 5
        mbart_config.d_model = 512
        mbart_config.encoder_attention_heads = 8
        mbart_config.decoder_attention_heads = 8
        mbart_config.decoder_ffn_dim = 2048
        mbart_config.decoder_ffn_dim = 2048
        model = CustomMBart(config=mbart_config)

        if not args.mbart:
            model.resize_token_embeddings(vocab_size + 2)
            model.init_weights()

        print(model)
        model.to(args.gpu)
        state_dict = torch.load(os.path.join(args.checkpoint_name_for_test, "best_model", "best_model.bin"))
        model.load_state_dict(state_dict)


        if args.mbart:
            idx = tokenizer.additional_special_tokens.index(LMAP[args.trg])
            special_ids = tokenizer.additional_special_tokens_ids
            trg_id = special_ids[idx]
            print(trg_id)

        else:
            src_id = len(tokenizer)
            trg_id = len(tokenizer)+1

        evaluator = NMTEvaluator(args, model, tokenizer=tokenizer, trg_id=trg_id)

        if not os.path.isdir(args.test_file):
            os.makedirs(args.test_file)

        output = evaluator.generate_epoch(test_gen)
        if args.beam:
            pd.to_pickle(output, os.path.join(args.test_file, "result-beam.pkl"))
        else:
            pd.to_pickle(output, os.path.join(args.test_file, "result-greedy.pkl"))

        # log_full_test_results_to_file(args, config=pretrained_config)


if __name__ == "__main__":

    args = InitialArgument()
    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.ngpus_per_node
    project_name = f"{args.src}-{args.trg}-scratch"

    wandb.init(project=project_name, name=f"{args.encoder_class}", reinit=True)
    wandb.config.update(args)

    if args.distributed_training:
        if args.ngpus_per_node < 2:
            raise ValueError("Require ngpu>=2")

        mp.spawn(run, nprocs=args.ngpus_per_node, args=(args,))
    else:
        run("cuda", args)
