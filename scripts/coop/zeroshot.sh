#!/bin/bash

# custom config
DATA=/root/autodl-tmp/dataset/tta_data
TRAINER=ZeroshotCLIP
DATASET=$1
CFG=$2  # rn50, rn101, vit_b32 or vit_b16

python CoOp_CoCoOp.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file ensemble/CoOp/configs/datasets/${DATASET}.yaml \
--config-file ensemble/CoOp/configs/trainers/CoOp/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--eval-only