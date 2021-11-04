
Data=ted # specify dataset
NGPU=6
CHECKPOINT=checkpoints # specify checkpoint
EC=facebook/mbart-large-50-many-to-many-mmt

S=20000

for lang in ko
do
  CUDA_VISIBLE_DEVICES=$NGPU python run_nmt.py \
    --dataset $Data \
    --root data \
    --encoder_class $EC \
    --do_test \
    --src en \
    --vocab_size $S \
    --trg $lang \
    --lr 3.0e-5 \
    --test_dir test_initial/$EC \
    --seed 1994 \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 8 \
    --gradient_accumulation_step 4 \
    --mixed_precision \
    --replace_vocab \
    --checkpoint_dir $CHECKPOINT \
    --checkpoint_name_for_test ./$CHECKPOINT/$EC-en-$lang-replace-$S ;
done