#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python tda_runner.py     --config ensemble/TDA/configs \
                                                --wandb-log \
                                                --datasets K \
                                                --data-root '/root/autodl-tmp/dataset/tta_data' \
                                                --backbone ViT-B/16