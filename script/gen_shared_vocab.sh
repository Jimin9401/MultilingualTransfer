#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH
# sciie chemprot citation_intent rct-20k

#dmis-lab/biobert-base-cased-v1.1 allenai/scibert_scivocab_uncased nfliu/scibert_basevocab_uncased

for lang in ko ja
do
python gen_custom_tokenizer.py \
  --root data \
  --src en \
  --trg $lang \
  --vocab_size 30000 ;
done