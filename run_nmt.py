from numpy.lib.arraysetops import isin
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
# import apex
from util.batch_generator import NMTBatchfier
from my_utils.tokenization_nmt import NMTTokenizer
from my_utils.nmt_dataset import ParallelDataset, ParallelDatasetV1, ParallelDatasetV2
from my_utils.label_smoother import LabelSmoother
from my_utils.bow_loss import BoWLoss
# from model.classification_model import PretrainedTransformer
from model.nmt_model import CustomMBart
import torch.multiprocessing as mp
from torch.utils.data import random_split
from util.data_builder import LMAP

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

def generate_parameters(encoder, except_group):
    for name, parameter in encoder.named_parameters():
        if name in except_group:
            continue
        else:
            yield parameter

def generate_combined_parameters(param1, param2):
    for param in param1:
        yield param
    for param in param2:
        yield param


def get_trainer(args, model, train_batchfier, test_batchfier, tokenizer):

    if args.initial_freeze:
        parameters_to_optimize = generate_combined_parameters(model.model.shared.parameters(), model.lm_head.parameters())
        # optimizer = torch.optim.AdamW(model.model.shared.parameters(), args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW(parameters_to_optimize, args.lr, weight_decay=args.weight_decay)
        zeroing_params = generate_parameters(model, 'model.shared.weight')
        zeroing_optimizer = torch.optim.AdamW(zeroing_params, args.lr, weight_decay=args.weight_decay)
        
    else:
        optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
        zeroing_optimizer=None

    # if args.mixed_precision:
    #     print('mixed_precision')
    #     scaler = torch.cuda.amp.GradScaler(enabled=True)
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
    if isinstance(tokenizer, dict):
        src_id = tokenizer['src'].lang_code
        trg_id = tokenizer['tgt'].lang_code
    else:
        src_id = tokenizer.convert_tokens_to_ids(LMAP[args.src])
        trg_id = tokenizer.convert_tokens_to_ids(LMAP[args.trg])

    print(args.src, src_id)
    print(args.trg, trg_id)

    if args.bow_loss:
        criteria = BoWLoss(epsilon=0.2, src_id=src_id, trg_id=trg_id)
    elif args.label_smoothing_factor != 0:
        criteria = LabelSmoother(epsilon=args.label_smoothing_factor)
    else:
        if args.replace_vocab_w_existing_lm_tokenizers:
            criteria = nn.CrossEntropyLoss(ignore_index=tokenizer['tgt'].padding_id)#ignore_index=1)
        else:
            criteria = nn.CrossEntropyLoss()#ignore_index=1)


    if args.max_train_steps is None:
        num_buckets = len(train_batchfier) // args.per_gpu_train_batch_size + (len(train_batchfier)%args.per_gpu_train_batch_size != 0)
        args.num_update_steps_per_epoch = num_buckets
        args.max_train_steps = args.n_epoch * args.num_update_steps_per_epoch
    
    print(f"max_train_steps: {args.max_train_steps}\n")

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    trainer = NMTTrainer(args, model, train_batchfier, test_batchfier, optimizer, args.gradient_accumulation_step,
                         criteria, args.clip_norm, args.mixed_precision, lr_scheduler, tokenizer, zeroing_optimizer=zeroing_optimizer)

    return trainer

def get_dataset(args, tokenizer):
    DATAPATH = args.datapath #/home/nas1_userD/yujinbaek/dataset/text/parallel_data/ted2020

    if args.train_dev_split:
        dataset = ParallelDataset(
            args,
            src_filename=os.path.join(DATAPATH,  f'train.{args.src}'),
            # src_filename=os.path.join(DATAPATH,  f'train.{args.src}-{args.trg}.{args.src}'),
            # tgt_filename=os.path.join(DATAPATH,  f'train.{args.src}-{args.trg}.{args.trg}'),
            tgt_filename=os.path.join(DATAPATH,  f'train.{args.trg}'),
            tokenizer=tokenizer,
        )
        split_ratio = [len(dataset)-args.validation_size, args.validation_size]
        train_dataset, dev_dataset = random_split(dataset, split_ratio)
    
    else:
        train_dataset = ParallelDataset(
            args,
            src_filename=os.path.join(DATAPATH,  f'train.{args.src}'),
            # src_filename=os.path.join(DATAPATH,  f'train.{args.src}-{args.trg}.{args.src}'),
            # tgt_filename=os.path.join(DATAPATH,  f'train.{args.src}-{args.trg}.{args.trg}'),
            tgt_filename=os.path.join(DATAPATH,  f'train.{args.trg}'),
            tokenizer=tokenizer,
        )

        dev_dataset = ParallelDataset(
            args,
            src_filename=os.path.join(DATAPATH,  f'dev.{args.src}'),
            # tgt_filename=os.path.join(DATAPATH,  f'dev.{args.src}-{args.trg}.{args.trg}'),
            # src_filename=os.path.join(DATAPATH,  f'dev.{args.src}-{args.trg}.{args.src}'),
            tgt_filename=os.path.join(DATAPATH,  f'dev.{args.trg}'),
            tokenizer=tokenizer,
        )

    test_dataset = ParallelDataset(
        args,
        src_filename=os.path.join(DATAPATH, f'test.{args.src}'),
        # tgt_filename=os.path.join(DATAPATH, f'test.{args.src}-{args.trg}.{args.trg}'),
        # src_filename=os.path.join(DATAPATH, f'test.{args.src}-{args.trg}.{args.src}'),
        tgt_filename=os.path.join(DATAPATH, f'test.{args.trg}'),
        tokenizer=tokenizer,
    )

    return train_dataset, dev_dataset, test_dataset

def get_datasetV1(args, tokenizer):
    DATAPATH = args.datapath #/home/nas1_userD/yujinbaek/dataset/text/parallel_data/ted2020


    train_dataset = ParallelDatasetV1(
        args,
        src_filename=os.path.join(DATAPATH,  f'train.{args.src}'),
        # src_filename=os.path.join(DATAPATH,  f'train.{args.src}-{args.trg}.{args.src}'),
        # tgt_filename=os.path.join(DATAPATH,  f'train.{args.src}-{args.trg}.{args.trg}'),
        tgt_filename=os.path.join(DATAPATH,  f'train.{args.trg}'),
        tokenizer=tokenizer,
    )

    dev_dataset = ParallelDatasetV1(
        args,
        src_filename=os.path.join(DATAPATH,  f'dev.{args.src}'),
        # tgt_filename=os.path.join(DATAPATH,  f'dev.{args.src}-{args.trg}.{args.trg}'),
        # src_filename=os.path.join(DATAPATH,  f'dev.{args.src}-{args.trg}.{args.src}'),
        tgt_filename=os.path.join(DATAPATH,  f'dev.{args.trg}'),
        tokenizer=tokenizer,
    )

    test_dataset = ParallelDatasetV1(
        args,
        src_filename=os.path.join(DATAPATH, f'test.{args.src}'),
        # tgt_filename=os.path.join(DATAPATH, f'test.{args.src}-{args.trg}.{args.trg}'),
        # src_filename=os.path.join(DATAPATH, f'test.{args.src}-{args.trg}.{args.src}'),
        tgt_filename=os.path.join(DATAPATH, f'test.{args.trg}'),
        tokenizer=tokenizer,
    )

    return train_dataset, dev_dataset, test_dataset

def get_datasetV2(args, tokenizer):
    DATAPATH = args.datapath #/home/nas1_userD/yujinbaek/dataset/text/parallel_data/ted2020


    train_dataset = ParallelDatasetV2(
        args,
        src_filename=os.path.join(DATAPATH,  f'train.{args.src}'),
        # src_filename=os.path.join(DATAPATH,  f'train.{args.src}-{args.trg}.{args.src}'),
        # tgt_filename=os.path.join(DATAPATH,  f'train.{args.src}-{args.trg}.{args.trg}'),
        tgt_filename=os.path.join(DATAPATH,  f'train.{args.trg}'),
        tokenizer=tokenizer,
    )

    dev_dataset = ParallelDatasetV2(
        args,
        src_filename=os.path.join(DATAPATH,  f'dev.{args.src}'),
        # tgt_filename=os.path.join(DATAPATH,  f'dev.{args.src}-{args.trg}.{args.trg}'),
        # src_filename=os.path.join(DATAPATH,  f'dev.{args.src}-{args.trg}.{args.src}'),
        tgt_filename=os.path.join(DATAPATH,  f'dev.{args.trg}'),
        tokenizer=tokenizer,
    )

    test_dataset = ParallelDatasetV2(
        args,
        src_filename=os.path.join(DATAPATH, f'test.{args.src}'),
        # tgt_filename=os.path.join(DATAPATH, f'test.{args.src}-{args.trg}.{args.trg}'),
        # src_filename=os.path.join(DATAPATH, f'test.{args.src}-{args.trg}.{args.src}'),
        tgt_filename=os.path.join(DATAPATH, f'test.{args.trg}'),
        tokenizer=tokenizer,
    )

    return train_dataset, dev_dataset, test_dataset

# def get_batchfier(args, tokenizer: MBart50Tokenizer):
#     n_gpu = torch.cuda.device_count()
#     train, dev, test = load_dataset(args, tokenizer, "nmt")
#     padding_idx = 1

#     train_batch = NMTBatchfier(args, train, batch_size=args.per_gpu_train_batch_size * n_gpu, maxlen=args.seq_len,
#                                padding_index=padding_idx, device="cuda")
#     dev_batch = NMTBatchfier(args, dev, batch_size=args.per_gpu_eval_batch_size * n_gpu, maxlen=args.seq_len,
#                              padding_index=padding_idx, device="cuda")
#     test_batch = NMTBatchfier(args, test, batch_size=args.per_gpu_eval_batch_size * n_gpu, maxlen=args.seq_len,
#                               padding_index=padding_idx, device="cuda")

#     return train_batch, dev_batch, test_batch


def run(gpu, args):
    # args = ExperimentArgument()
    args.gpu = gpu
    args.device = gpu
    args.aug_ratio = 0.0
    set_seed(args.seed)

    print(args.__dict__)

    pretrained_config = MBartConfig.from_pretrained(args.encoder_class)

    pretrained_config.dropout = args.dropout
    pretrained_config.attention_dropout = args.attention_dropout
    pretrained_config.activation_dropout = args.activation_dropout

    from util.data_builder import LMAP


    model = CustomMBart.from_pretrained(args.encoder_class, cache_dir='/home/nas1_userD/yujinbaek/huggingface')

    # dropout settings
    for (module_name, module) in model.named_modules():
        if hasattr(module, 'dropout'):
            if 'attn' in module_name:
                module.dropout = args.attention_dropout
            else:
                module.dropout = args.dropout
        if hasattr(module, 'activation_dropout'):
            module.activation_dropout = args.activation_dropout

    # if args.embedding_adapting_process:
    #     from copy import deepcopy
    #     target_embedding = deepcopy(model.model.encoder.embed_tokens)
    # else:
    #     target_embedding = None


    if args.replace_vocab:
        from tokenizers import SentencePieceBPETokenizer
        vocab_size=args.vocab_size
        logger.info(f"check vocab size::: currently you are using {vocab_size}")
        
        if args.corpora_bpe:
            vocab_json_name = args.vocabpath + f"/{args.src}{args.trg}-{vocab_size}-corpora-vocab.json"
            merge_name = args.vocabpath + f"/{args.src}{args.trg}-{vocab_size}-corpora-merges.txt"
        else:
            vocab_json_name = args.vocabpath + f"/{args.src}{args.trg}-{vocab_size}-vocab.json"
            merge_name = args.vocabpath + f"/{args.src}{args.trg}-{vocab_size}-merges.txt"

        print(vocab_json_name)
        print(merge_name)

        src_lang=LMAP[args.src]
        tgt_lang=LMAP[args.trg]
        additional_special_tokens=[src_lang, tgt_lang]

        deployed_tokenizer = NMTTokenizer(vocab_filename=vocab_json_name, merges_filename=merge_name, src_lang=src_lang, tgt_lang=tgt_lang, additional_special_tokens=additional_special_tokens)
        
        pretrained_tokenizer = MBart50Tokenizer.from_pretrained(args.encoder_class, src_lang=LMAP[args.src],
                                                                tgt_lang=LMAP[args.trg], cache_dir='/home/nas1_userD/yujinbaek/huggingface')

        from corpus_utils.vocab_util import align_vocabularies, align_vocabularies_with_dictionary

        if args.bilingual_dictionary_filename:
            new_dict = align_vocabularies_with_dictionary(pretrained_tokenizer, deployed_tokenizer, args.bilingual_dictionary_filename)
            
        else:
            new_dict = align_vocabularies(pretrained_tokenizer, deployed_tokenizer)

        # special_map = {k: v for k, v in
        #                zip(pretrained_tokenizer.additional_special_tokens,
        #                    pretrained_tokenizer.additional_special_tokens_ids)}
        # special_ids = [special_map[LMAP[args.src]], special_map[LMAP[args.trg]]]
        # args.new_special_src_id = len(new_dict)
        # args.new_special_trg_id = len(new_dict) + 1
        model.rearrange_token_embedding(new_dict)#, special_ids)

        if args.random_embeddings:
            model._init_weights(model.model.encoder.embed_tokens)
            randomized_weights = model.model.encoder.embed_tokens.weight.data
            model.lm_head.weight.data = randomized_weights

        # if args.embedding_adapting_process:
            # [3,4] -> [2,5,6]

        if args.vr_with_adapted_embeddings:#
            data = torch.load(args.adapted_embeddings_filepath)
            model.model.encoder.embed_tokens.load_state_dict(data)
            # check whether encoder embeddings + decoder embeddings + lm_head are tied correctly
            not_coupled = (model.model.encoder.embed_tokens.weight.data == model.model.decoder.embed_tokens.weight.data).ne(1).long().sum().item()
            if not_coupled:
                print("encoder and decoder embeddings are not coupled correctly")
            not_coupled = (model.model.encoder.embed_tokens.weight.data == model.lm_head.weight.data).ne(1).long().sum().item()
            if not_coupled:
                print("encoder embedding and decoder lm head are not coupled correctly")
            print(f"embedding weights are properly loaded from {args.adapted_embeddings_filepath}")

    elif args.replace_vocab_w_existing_lm_tokenizers:
        from transformers import AutoTokenizer, BertTokenizerFast, BartTokenizerFast
        from corpus_utils.vocab_util import align_vocabularies, align_vocabularies_w_bert_tokenizer, align_vocabularies_w_bart_tokenizer

        src_tokenizer = AutoTokenizer.from_pretrained(args.src_tokenizer, cache_dir='/home/nas1_userD/yujinbaek/huggingface')
        tgt_tokenizer = AutoTokenizer.from_pretrained(args.tgt_tokenizer, cache_dir='/home/nas1_userD/yujinbaek/huggingface')

        pretrained_tokenizer = MBart50Tokenizer.from_pretrained(
            args.encoder_class, 
            src_lang=LMAP[args.src],
            tgt_lang=LMAP[args.trg]
            )

        if isinstance(src_tokenizer, BertTokenizerFast):
            src_new_dict = align_vocabularies_w_bert_tokenizer(pretrained_tokenizer, src_tokenizer, language=args.src)
        else:
            src_new_dict = align_vocabularies_w_bart_tokenizer(pretrained_tokenizer, src_tokenizer, language=args.src)

        if isinstance(tgt_tokenizer, BertTokenizerFast):
            tgt_new_dict = align_vocabularies_w_bert_tokenizer(pretrained_tokenizer, tgt_tokenizer, language=args.trg)
        else:
            tgt_new_dict = align_vocabularies_w_bart_tokenizer(pretrained_tokenizer, tgt_tokenizer, language=args.trg)
        
        new_dict = {'src': src_new_dict, 'tgt': tgt_new_dict}
        deployed_tokenizer = {'src': src_tokenizer, 'tgt': tgt_tokenizer}
        model.rearrange_token_embedding_w_existing_lm_tokenizers(new_dict)
        
        # tying special tokens
        ## share special tokens btw encoder and decoder
        # </s> [SEP]
        # decoder_eos_token_id = tgt_tokenizer.eos_token_id
        # encoder_eos_token_id = src_tokenizer.eos_token_id if src_tokenizer.eos_token_id is not None else src_tokenizer.sep_token_id
        # model.lm_head.weight[decoder_eos_token_id] = model.model.encoder.embed_tokens.weight[encoder_eos_token_id]
        # # PAD
        # decoder_pad_token_id = tgt_tokenizer.pad_token_id
        # encoder_pad_token_id = src_tokenizer.pad_token_id
        # model.lm_head.weight[decoder_pad_token_id] = model.model.encoder.embed_tokens.weight[encoder_pad_token_id]

    else:
        special_ids = [LMAP[args.src], LMAP[args.trg]]
        deployed_tokenizer = MBart50Tokenizer.from_pretrained(
            args.encoder_class, 
            src_lang=LMAP[args.src],
            tgt_lang=LMAP[args.trg], 
            cache_dir='/home/nas1_userD/yujinbaek/huggingface'
            )

    args.extended_vocab_size = 0
    if isinstance(deployed_tokenizer, dict):
        train_gen, dev_gen, test_gen = get_datasetV1(args, deployed_tokenizer)
        # train_gen, dev_gen, test_gen = get_datasetV2(args, deployed_tokenizer)
    else:
        train_gen, dev_gen, test_gen = get_dataset(args, deployed_tokenizer)

    model.to("cuda")
    wandb.watch(model)
    trainer = get_trainer(args, model, train_gen, dev_gen, deployed_tokenizer)
    best_dir = os.path.join(args.savename, "best_model")

    if not os.path.isdir(best_dir):
        os.makedirs(best_dir)

    results = []

    optimal_bleu = 0.0
    not_improved = 0

    if args.do_train:
        for e in tqdm(range(0, args.n_epoch)):
            print("Epoch : {0}".format(e))

            state = (model.model.decoder.embed_tokens.weight.data == model.lm_head.weight.data)            
            print("STATE: ", state.long().ne(1).sum())
            trainer.train_epoch()

            if args.evaluate_during_training:
                eval_metric = trainer.test_epoch(epoch=e)
                results.append({"eval_bleu": eval_metric["score"]})

                if optimal_bleu < eval_metric["score"]:
                    optimal_bleu = eval_metric["score"]
                    # if args.distributed_training:
                    #     torch.save(model.module.state_dict(), os.path.join(best_dir, "best_model.bin"))
                    # else:
                    torch.save(model.state_dict(), os.path.join(best_dir, f"best_model_{args.wandb_run_name}.bin"))
                    print("Update Model checkpoints at {0}!! ".format(best_dir))
                    not_improved = 0
                else:
                    not_improved += 1
            
            # if not_improved >= 5:
            #     break

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
        model = CustomMBart.from_pretrained(args.encoder_class, cache_dir='/home/nas1_userD/yujinbaek/huggingface')
        if args.replace_vocab:
            # model.resize_token_embeddings(len(new_dict) + 2)
            if 'tgt' in new_dict:
                new_dict = new_dict['tgt']
            model.resize_token_embeddings(len(new_dict))
            model.final_logits_bias.data = torch.zeros([1, 250054])

        model.to(args.gpu)
        # state_dict = torch.load(os.path.join(args.checkpoint_name_for_test, "epoch_4", "epoch_4_model.bin"))
        if args.vr_with_adapted_embeddings:
            print("zero shot test")
        elif args.replace_vocab:
            state_dict = torch.load(os.path.join(args.checkpoint_name_for_test, "best_model", f"best_model_{args.wandb_run_name}.bin"))
            model.load_state_dict(state_dict)
            shape = model.final_logits_bias.data.shape
            model.final_logits_bias.data = torch.zeros(shape)  # we remove final logit bias when training
        else:
            filename = os.path.join(args.checkpoint_name_for_test, "best_model", f"best_model_{args.wandb_run_name}.bin")
            print(filename)
            state_dict = torch.load(os.path.join(args.checkpoint_name_for_test, "best_model", f"best_model_{args.wandb_run_name}.bin"))
            model.load_state_dict(state_dict)

        if args.replace_vocab:
            # trg_id = len(new_dict) + 1
            trg_id = deployed_tokenizer.convert_tokens_to_ids(LMAP[args.trg])
        else:
            idx=deployed_tokenizer.additional_special_tokens.index(LMAP[args.trg])
            special_ids = deployed_tokenizer.additional_special_tokens_ids
            trg_id =special_ids[idx]

        evaluator = NMTEvaluator(args, model, tokenizer=deployed_tokenizer, trg_id=trg_id)

        # if not os.path.isdir(args.test_file):
        #     os.makedirs(args.test_file)

        # output = evaluator.generate_epoch(test_gen)
        score = evaluator.generate_epoch(test_gen)
        wandb.log({'test_score': score})

        # if args.beam:
        #     pd.to_pickle(output, os.path.join(args.test_file, "result-beam.pkl"))
        # else:
        #     pd.to_pickle(output, os.path.join(args.test_file, "result-greedy.pkl"))

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
    
    if args.encoder_class.endswith("one-to-many-mmt"):
        project_name+="-one_to_many_mmt"

    elif args.encoder_class.endswith("many-to-many-mmt"):
        project_name+="-many_to_many_mmt"


    wandb.init(project=project_name, reinit=True, name=args.wandb_run_name)

    if args.wandb_run_name:
        wandb.run.name =  args.wandb_run_name
        wandb.run.save()
    else:
        args.wandb_run_name = wandb.run.name

    wandb.config.update(args)

    if args.distributed_training:
        if args.ngpus_per_node < 2:
            raise ValueError("Require ngpu>=2")

        mp.spawn(run, nprocs=args.ngpus_per_node, args=(args,))
    else:
        run("cuda", args)
