
Data=ted # specify dataset
NGPU=3
CHECKPOINT=./initial # specify checkpoint
EC=facebook/mbart-large-50
#EC=facebook/mbart-large-50-many-to-many-mmt

#for lang in tr
#do
#CUDA_VISIBLE_DEVICES=$NGPU python run_preliminary_results.py \
#  --dataset $Data \
#  --root data \
#  --do_train \
#  --src en \
#  --trg $lang \
#  --vocab_size 30000 \
#  --evaluate_during_training \
#  --lr 5.0e-5 \
#  --seed 1994 \
#  --mbart \
#  --per_gpu_train_batch_size 24 \
#  --per_gpu_eval_batch_size 8 \
#  --gradient_accumulation_step 4 \
#  --mixed_precision \
#  --num_warmup_steps 2500 \
#  --encoder_class $EC \
#  --n_epoch 90 \
#  --checkpoint_dir $CHECKPOINT ;
#done


for lang in tr
do
CUDA_VISIBLE_DEVICES=$NGPU python run_preliminary_results.py \
  --root dataset \
  --dataset $Data \
  --do_train \
  --src en \
  --trg $lang \
  --vocab_size 30000 \
  --evaluate_during_training \
  --lr 5.0e-5 \
  --seed 1994 \
  --num_warmup_steps 2500 \
  --per_gpu_train_batch_size 96 \
  --per_gpu_eval_batch_size 8 \
  --gradient_accumulation_step 1 \
  --mixed_precision \
  --encoder_class $EC \
  --n_epoch 100 \
  --checkpoint_dir $CHECKPOINT ;
done


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