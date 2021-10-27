
Data=ted # specify dataset
NGPU=1
CHECKPOINT=./checkpoints_new # specify checkpoint

#CUDA_VISIBLE_DEVICES=$NGPU python run_nmt.py \
#  --dataset $Data \
#  --root data \
#  --do_test \
#  --src en \
#  --trg ja \
#  --lr 5.0e-5 \
#  --seed 1994 \
#  --per_gpu_train_batch_size 4 \
#  --per_gpu_eval_batch_size 8 \
#  --gradient_accumulation_step 4 \
#  --mixed_precision \
#  --replace_vocab \
#  --checkpoint_dir $CHECKPOINT \
#  --checkpoint_name_for_test ./checkpoints/nmt-en-ja-replace ;


#for lang in fi
#do
#  CUDA_VISIBLE_DEVICES=$NGPU python run_nmt.py \
#    --dataset $Data \
#    --root data \
#    --do_test \
#    --src en \
#    --trg $lang \
#    --lr 5.0e-5 \
#    --seed 1994 \
#    --per_gpu_train_batch_size 4 \
#    --per_gpu_eval_batch_size 8 \
#    --beam \
#    --gradient_accumulation_step 4 \
#    --mixed_precision \
#    --checkpoint_dir $CHECKPOINT \
#    --replace_vocab \
#    --checkpoint_name_for_test ./checkpoints_new/nmt-en-$lang-replace ;
#done


for lang in ko ja
do
  CUDA_VISIBLE_DEVICES=$NGPU python run_nmt.py \
    --dataset $Data \
    --root data \
    --do_test \
    --src en \
    --trg $lang \
    --lr 5.0e-5 \
    --test_dir test_2 \
    --seed 1994 \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 8 \
    --gradient_accumulation_step 4 \
    --mixed_precision \
    --checkpoint_dir $CHECKPOINT \
    --replace_vocab \
    --checkpoint_name_for_test ./checkpoints_fix/nmt-en-$lang-replace;
done