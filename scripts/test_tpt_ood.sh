#!/bin/bash

data_root='/root/autodl-tmp/dataset/tta_data'
testsets=I
# I/A/V/R/K
#testsets=$1
#arch=RN50
arch=ViT-B/16
bs=64
ctx_init=a_photo_of_a
run_type=tpt

python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu 0 \
--tpt --ctx_init ${ctx_init} --run_type ${run_type} --I_augmix \
