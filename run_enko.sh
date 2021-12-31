DEVICE_NUMBER=6
SRC=en
TRG=ko
CHECKPOINT_DIR=nmt_${SRC}${TRG}_50k
ENCODER_CLASS=facebook/mbart-large-50-one-to-many-mmt
DATAPATH=/home/nas1_userD/yujinbaek/dataset/text/parallel_data/ted2020/train-test
VOCABPATH=/home/nas1_userD/yujinbaek/dataset/text/parallel_data/en-ko/vocab-train/en-ko-shared-w-mecab
VOCAB_SIZE=50000
MAX_LR=3.0e-5
LR_SCHEDULER_TYPE=linear
NUM_WARMUP_STEPS=5000
N_EPOCH=8
GRADIENT_ACCUMULATION_STEP=3
SEED=1994
LABEL_SMOOTHING_FACTOR=0.2
DROPOUT=0.3
ACTIVATION_DROPOUT=0.1
ATTENTION_DROPOUT=0.1
PER_GPU_TRAIN_BATCH_SIZE=8
PER_GPU_EVAL_BATCH_SIZE=8
REPLACE_VOCAB=-replace
WANDB_RUN_NAME='abc-def-202'

CUDA_VISIBLE_DEVICES=${DEVICE_NUMBER} OMP_NUM_THREADS=1 python run_nmt.py --dataset ted --datapath ${DATAPATH} --replace_vocab --do_train --evaluate_during_training --beam --encoder_class ${ENCODER_CLASS} --vocab_size ${VOCAB_SIZE} --src ${SRC} --trg ${TRG} --n_epoch ${N_EPOCH} --gradient_accumulation_step ${GRADIENT_ACCUMULATION_STEP} --per_gpu_train_batch_size ${PER_GPU_TRAIN_BATCH_SIZE} --per_gpu_eval_batch_size ${PER_GPU_EVAL_BATCH_SIZE} --lr ${MAX_LR} --lr_scheduler_type ${LR_SCHEDULER_TYPE} --num_warmup_steps ${NUM_WARMUP_STEPS} --seed ${SEED} --checkpoint_dir /home/nas1_userD/yujinbaek/out/${CHECKPOINT_DIR} --vocabpath ${VOCABPATH} --test_dir /home/nas1_userD/yujinbaek/out/nmt_${SRC}${TRG} --wandb_run_name ${WANDB_RUN_NAME} --label_smoothing_factor ${LABEL_SMOOTHING_FACTOR} --dropout ${DROPOUT} --activation_dropout ${ACTIVATION_DROPOUT} --attention_dropout ${ATTENTION_DROPOUT}


echo 'Training is DONE!'
echo 'Do Evaluation ::::::'

CUDA_VISIBLE_DEVICES=${DEVICE_NUMBER} OMP_NUM_THREADS=1 python run_nmt.py \
--replace_vocab --do_test --beam \
--vocab_size ${VOCAB_SIZE} --src ${SRC} --trg ${TRG} \
--per_gpu_eval_batch_size ${PER_GPU_EVAL_BATCH_SIZE} \
--dataset ted --datapath ${DATAPATH} \
--vocabpath ${VOCABPATH} --encoder_class ${ENCODER_CLASS} \
--checkpoint_name_for_test /home/nas1_userD/yujinbaek/out/${CHECKPOINT_DIR}/${ENCODER_CLASS}-${SRC}-${TRG}${REPLACE_VOCAB} \
--wandb_run_name ${WANDB_RUN_NAME}\

