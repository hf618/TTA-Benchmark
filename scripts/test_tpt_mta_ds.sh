#!/bin/bash

data_root=/root/autodl-tmp/dataset/tta_data
testsets=$1
#arch=RN50
arch=ViT-B/16
bs=64
ctx_init=a_photo_of_a

lambda_q=4
lambda_y=0.2
python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu 0 \
--tpt --ctx_init ${ctx_init} \
--mta --lambda_q ${lambda_q} --lambda_y ${lambda_y} \