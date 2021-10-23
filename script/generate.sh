
EC=gpt2 # specify encoder
Data=rct-20k # specify dataset
NGPU=4
CHECKPOINT=./checkpoints # specify checkpoint

CUDA_VISIBLE_DEVICES=$NGPU python run_generator.py \
  --dataset $Data \
  --root data \
  --do_test \
  --lr 5.0e-5 \
  --seed_list 1994 \
  --per_gpu_eval_batch_size 20 \
  --beam \
  --top_whatever 10 \
  --nprefix 50 \
  --vocab_size 50257 \
  --ngenerate 50 \
  --mixed_precision \
  --checkpoint_dir $CHECKPOINT \
  --encoder_class $EC;


CUDA_VISIBLE_DEVICES=$NGPU python run_generator.py \
  --dataset $Data \
  --root data \
  --do_test \
  --lr 5.0e-5 \
  --seed_list 1994 \
  --per_gpu_eval_batch_size 20 \
  --beam \
  --init_embed \
  --top_whatever 10 \
  --nprefix 50 \
  --vocab_size 50257 \
  --ngenerate 50 \
  --mixed_precision \
  --checkpoint_dir $CHECKPOINT \
  --encoder_class $EC;