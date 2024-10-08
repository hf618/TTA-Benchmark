#!/bin/bash
#cd ../..

# custom config
DATA='/root/autodl-tmp/dataset/tta_data' 
TRAINER=PromptAlign
# imagenet/imagenet_a/imagenet_r/imagenet_sketch/imagenetv2
# caltech101/dtd/eurosat/fgvc_aircraft/food101/oxford_flowers/oxford_pets/stanford_cars/sun397/ucf101
DATASET=imagenet # 往yaml走
SEED=1
CUSTOM_NAME=$1

WEIGHTSPATH='/root/autodl-tmp/pretrained/MaPLe/cross-domain-datasets/imagenet'

CFG=DG_PAlign_vit_b16_c2_ep5_batch4_2ctx_cross_datasets
SHOTS=16
LOADEP=2

MODEL_DIR=${WEIGHTSPATH}/seed${SEED}

DIR=output/${TRAINER}/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are already available in ${DIR}. Skipping..."
else
    echo "Evaluating model"
    echo "Runing the first phase job and save the output to ${DIR}"
    # Evaluate on evaluation datasets
    python main_pa.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file ensemble/PA/configs/datasets/${DATASET}.yaml \
    --config-file ensemble/PA/configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --tpt \
    DATASET.NUM_SHOTS ${SHOTS} \

fi