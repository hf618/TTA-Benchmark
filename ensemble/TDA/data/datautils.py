import os
from typing import Tuple
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data.hoi_dataset import BongardDataset
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.fewshot_datasets import *
import data.augmix_ops as augmentations

import ipdb
# .sh按照keys来，注意cross数据集则按照fewshot_datasets.py
# 实际文件夹命名按照values来
ID_to_DIRNAME={
    'I': 'ImageNet',
    'A': 'imagenet-a',
    'K': 'ImageNet-Sketch',
    'R': 'imagenet-r',
    'V': 'imagenetv2-matched-frequency-format-val',
    'flower102': 'Flower102',
    'dtd': 'DTD',
    'pets': 'OxfordPets',
    'cars': 'StanfordCars',
    'ucf101': 'UCF101',
    'caltech101': 'Caltech101',
    'food101': 'Food101',
    'sun397': 'SUN397',
    'aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat'
}

distortions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                'defocus_blur', 'glass_blur',
                'zoom_blur', 'frost',
                'brightness', 'contrast', 'elastic_transform',
                'pixelate','fog','speckle_noise','saturate', 'spatter', 'gaussian_blur']


def build_dataset(set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False, num_classes=None):
    if set_id == 'I':
        # ImageNet validation set
        testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'val')
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in ['A', 'K', 'R', 'V']:
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in fewshot_datasets:
        if mode == 'train' and n_shot:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode)

    elif set_id == 'bongard':
        assert isinstance(transform, Tuple)
        base_transform, query_transform = transform
        testset = BongardDataset(data_root, split, mode, base_transform, query_transform, bongard_anno)
    else:
        raise NotImplementedError
        
    return testset


# # AugMix Transforms
# def get_preaugment():
#     return transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#         ])
# AugMix Transforms
def get_preaugment(hard_aug=False, resolution=224, crop_min=0.2):
    if hard_aug:
        # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
        return transforms.Compose([
            transforms.RandomResizedCrop(resolution, scale=(crop_min, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.1),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            transforms.RandomHorizontalFlip(),
            ])
    else:
        return transforms.Compose([
                transforms.RandomResizedCrop(resolution),
                transforms.RandomHorizontalFlip(),
            ])

def augmix(image, preaugment, preprocess, aug_list, severity=1):
    # preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1, hard_aug=False):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        self.aug_list = augmentations.augmentations if augmix else []
        self.severity = severity
        self.preaugment = get_preaugment(hard_aug=hard_aug, resolution=224, crop_min=0.2)
        print("\n AugMixAugmenter created: \n"
                "\t len(aug_list): {}, augmix: {} \n".format(len(self.aug_list), augmix))
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preaugment, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views



