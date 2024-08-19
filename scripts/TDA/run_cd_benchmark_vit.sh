#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python tda_runner.py     --config configs \
                                                --wandb-log \
                                                --datasets stanford_cars \
                                                --data-root '/root/autodl-tmp/dataset/tta_data' \
                                                --backbone ViT-B/16
                                              # caltech101/dtd/eurosat/fgvc/food101/oxford_flowers/oxford_pets/stanford_cars/sun397/ucf101