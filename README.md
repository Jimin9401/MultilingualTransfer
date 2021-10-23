execute with run_nmt.py file

````bash
Data=ted # specify dataset
NGPU=1
CHECKPOINT=./checkpoints # specify checkpoint

CUDA_VISIBLE_DEVICES=$NGPU python run_nmt.py \
  --dataset $Data \
  --root data \
  --do_train \
  --src en \
  --trg ko \
  --evaluate_during_training \
  --lr 5.0e-5 \
  --seed 1994 \
  --per_gpu_train_batch_size 4 \
  --per_gpu_eval_batch_size 8 \
  --gradient_accumulation_step 4 \
  --mixed_precision \
  --n_epoch 10 \
  --checkpoint_dir $CHECKPOINT ;

````

