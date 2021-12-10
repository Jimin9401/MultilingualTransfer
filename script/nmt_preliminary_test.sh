
Data=ted # specify dataset
NGPU=0
CHECKPOINT=./preliminary_shared # specify checkpoint
EC=facebook/mbart-large-50
#EC=facebook/mbart-large-50-many-to-many-mmt


for lang in es ko ja
do
  CUDA_VISIBLE_DEVICES=$NGPU python run_preliminary_results.py \
    --dataset $Data \
    --root data \
    --encoder_class $EC \
    --do_test \
    --src en \
    --trg $lang \
    --lr 3.0e-5 \
    --test_dir scratch/scratch-mbart \
    --seed 1994 \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 64 \
    --gradient_accumulation_step 4 \
    --mixed_precision \
    --mbart \
    --beam \
    --checkpoint_dir $CHECKPOINT \
    --checkpoint_name_for_test ./initial/scratch-en-$lang-mbart ;

#  CUDA_VISIBLE_DEVICES=$NGPU python run_preliminary_results.py \
#    --dataset $Data \
#    --root dataset \
#    --encoder_class $EC \
#    --do_test \
#    --src en \
#    --trg $lang \
#    --lr 3.0e-5 \
#    --test_dir scratch/scratch-mono \
#    --seed 1994 \
#    --per_gpu_train_batch_size 4 \
#    --per_gpu_eval_batch_size 64 \
#    --gradient_accumulation_step 4 \
#    --mixed_precision \
#    --checkpoint_dir $CHECKPOINT \
#    --beam \
#    --checkpoint_name_for_test ./preliminary_shared/scratch-en-$lang-mono
done
