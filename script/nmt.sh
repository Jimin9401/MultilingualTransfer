
Data=ted # specify dataset
NGPU=7
CHECKPOINT=./checkpoints # specify checkpoint

CUDA_VISIBLE_DEVICES=$NGPU python run_nmt.py \
  --dataset $Data \
  --root data \
  --do_train \
  --src en \
  --trg fi \
  --evaluate_during_training \
  --lr 5.0e-5 \
  --seed 1994 \
  --per_gpu_train_batch_size 4 \
  --per_gpu_eval_batch_size 8 \
  --gradient_accumulation_step 4 \
  --mixed_precision \
  --n_epoch 10 \
  --replace_vocab \
  --checkpoint_dir $CHECKPOINT ;


#CUDA_VISIBLE_DEVICES=$NGPU python run_nmt.py \
#  --dataset $Data \
#  --root data \
#  --do_train \
#  --src en \
#  --trg tr \
#  --evaluate_during_training \
#  --lr 5.0e-5 \
#  --seed 1994 \
#  --per_gpu_train_batch_size 4 \
#  --per_gpu_eval_batch_size 8 \
#  --gradient_accumulation_step 4 \
#  --mixed_precision \
#  --n_epoch 10 \
#  --checkpoint_dir $CHECKPOINT ;



  #  --n_sample None \
#  --encoder_class $EC;


#CUDA_VISIBLE_DEVICES=$NGPU python run_nmt.py \
#  --dataset $Data \
#  --root data \
#  --do_test \
#  --lr 5.0e-5 \
#  --seed_list 1994 \
#  --per_gpu_eval_batch_size 20 \
#  --beam \
#  --init_embed \
#  --top_whatever 10 \
#  --nprefix 50 \
#  --vocab_size 50257 \
#  --ngenerate 50 \
#  --mixed_precision \
#  --checkpoint_dir $CHECKPOINT \
#  --encoder_class $EC;