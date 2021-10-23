#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH
# sciie chemprot citation_intent rct-20k

#dmis-lab/biobert-base-cased-v1.1 allenai/scibert_scivocab_uncased nfliu/scibert_basevocab_uncased

EC=gpt2
V=50257
Data=amazon

python reassemble.py --dataset $Data \
  --root data \
  --vocab_size $V \
  --use_fragment \
  --encoder_class $EC;
