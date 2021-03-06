CUDA_VISIBLE_DEVICES=$NGPU python corpuswise_eval.py \
  --dataset $Data \
  --root data \
  --do_test \
  --lr 5.0e-5 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 256 \
  --mixed_precision \
  --evaluate_during_training \
  --checkpoint_dir $CHECKPOINT \
  --test_log_dir=test_log \
  --encoder_class $EC;
