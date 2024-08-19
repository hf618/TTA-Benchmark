#!/bin/bash


### Remember to change root to /YOUR/PATH ###
root=/root/autodl-tmp
#############################################

data_root=${root}/dataset/tta_data
# Dataset
testsets=$1

# Config
arch=ViT-B/16
coop_weight=${root}/pretrained/coop/coop_16shots_nctx4_cscFalse_ctpend_vitb16_seed1/model.pth.tar-50
ctx_init=a_photo_of_a
tta_steps=3
lr=7e-3
weight_decay=5e-4

# augmentation views and selection ratio
batch_size=64
selection_p=0.1

# Config for CLIP reward
reward_arch=ViT-L/14
reward_amplify=0
reward_process=1
process_batch=0
sample_k=3


runfile=${root}/AHL-TPT/tpt_cls_rl.py

# 通过一个 for 循环，迭代不同的配置编号 num（在这里只有 01）。
# 根据 num 的值，调整一些参数，例如 TTA 步数、学习率和测试数据集。
for num in 01
do
    case ${num} in
        01 )
            tta_steps=3
            lr=7e-3
            reward_arch=ViT-L/14
            testsets=$1
            ;;
        * )
            ;;
    esac
# 置输出目录 output，它基于 root 路径和 rlcf_prompt_${num} 命名模式构建。
output=${root}/output/RLCF/rlcf_prompt_${num}

python ${runfile} ${data_root} \
        --test_sets ${testsets} \
        -a ${arch} \
        --batch_size ${batch_size} \
        --selection_p ${selection_p} \
        --gpu 0 \
        --tpt \
        --ctx_init ${ctx_init} \
        --tta_steps ${tta_steps} \
        --lr ${lr} \
        --weight_decay ${weight_decay} \
        --output ${output} \
        --load ${coop_weight} \
        --reward_amplify ${reward_amplify} \
        --reward_process ${reward_process} \
        --process_batch ${process_batch} \
        --reward_arch ${reward_arch} \
        --sample_k ${sample_k}

done

echo "ghost cupbearer tinsmith richly automatic rewash liftoff ripcord april fruit voter resent facebook" >/dev/null
