#!/bin/bash

# parameters for the script
SET_ID="cars" # dataset identifier, have a look at data/datautils.py
RESULTS_DIR=results_dev # folder where results are saved
CTX="a_photo_of_a" # underscore separated hard prompt for CLIP (I've always used this one!)
SEED=1 # random seed (can only be 1,2 or 3 for MaPLe, any number will work otherwise)

# launch :)
python zero_classification.py \
--tuner ZeroRLCF \
--arch ViT-B/16 \
--reward_model ViT-L/14 \
--ctx_init ${CTX} \
--set_id ${SET_ID} \
--results_dir ${RESULTS_DIR} \
--seed ${SEED}