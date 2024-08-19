#!/bin/bash

# custom config
DATA=/root/autodl-tmp/dataset/tta_data
TRAINER=CoOp
SHOTS=16
NCTX=16
CSC=False
CTP=end

DATASET=$1
CFG=$2

for SEED in 1 2 3
do
    python CoOp_CoCoOp.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file ensemble/CoOp/configs/datasets/${DATASET}.yaml \
    --config-file ensemble/CoOp/configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir output/${TRAINER}/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/${DATASET}/seed${SEED} \
    --model-dir output/${TRAINER}/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED} \
    --load-epoch 50 \
    --eval-only \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP}
done