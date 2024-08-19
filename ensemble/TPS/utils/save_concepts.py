import argparse
import os
import json
import torch
import torch.nn.functional as F
import pickle
import numpy as np

from tqdm import tqdm

import sys
sys.path.append(os.getcwd())
from ensemble.TPS.model import load, tokenize
from data.imagenet_prompts_clean import imagenet_classes, imagenet_templates
from data.cls_to_names import (
    flower102_classes, 
    food101_classes,
    dtd_classes,
    pets_classes,
    ucf101_classes,
    aircraft_classes,
    eurosat_classes,
    sun397_classes,
    caltech101_classes,
    cars_classes
)

CLASSES_DICT = {
    "aircraft": aircraft_classes,
    "DTD": dtd_classes,
    "dtd": dtd_classes,
    "flowers": flower102_classes,
    "flower": flower102_classes,
    "food101": food101_classes,
    "food": food101_classes,
    "UCF101": ucf101_classes,
    "ucf101": ucf101_classes,
    "pets": pets_classes,
    "EuroSAT": eurosat_classes,
    "eurosat-new": eurosat_classes,
    "ImageNet": imagenet_classes,
    "SUN397": sun397_classes,
    "sun397": sun397_classes,
    "CalTech101": caltech101_classes,
    "caltech101": caltech101_classes,
    "cars": cars_classes,
}

device='cuda'
DOWNLOAD_ROOT='~/.cache/clip/'

def make_descriptor_sentence(descriptor):
# Code from https://github.com/sachit-menon/classify_by_description_release/blob/master/descriptor_strings.py#L43
    if descriptor.startswith('a') or descriptor.startswith('an'):
        return f"which is {descriptor}"
    elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
        return f"which {descriptor}"
    elif descriptor.startswith('used'):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"


def save_concepts(args):
    clip, _, _ = load(args.arch, device=device, download_root=DOWNLOAD_ROOT)

    # 如果命令行参数--no_cond被提供，suffix变量被设置为"_no_cond"。
    # 这意味着生成的嵌入文件将保存在包含这个后缀的目录中，例如"ViT-B/16_embeds_no_cond"。
    # 这个后缀可能用于标识那些没有使用任何特定条件或模板生成的嵌入。
    if args.no_cond:
        suffix = "_no_cond"
    elif args.x_templates:
        suffix = "_x_templates"
    else:
        suffix = ""
    # 构造用于存储生成的概念嵌入（concept embeddings）的目录路径。
    gpt4_concepts_embeds_dir = os.path.join(args.arch.replace('/', '-').lower() + "_embeds", args.gpt4_concepts_embeds_dir + suffix)

    os.makedirs(gpt4_concepts_embeds_dir, exist_ok=True)
    # 从命令行参数中获取概念的JSON文件路径，这个文件包含了用于生成文本提示的概念或描述符。
    concepts_json = args.concepts_json
    # 从命令行参数中获取当前处理的数据集名称，例如'ImageNet'、'food101'等。
    dataset = args.dataset

    concept_embeds_path = os.path.join(gpt4_concepts_embeds_dir, f'{dataset}.pkl')

    print(f"Saving concept embeds to {concept_embeds_path}")

    if args.save_concept_dict: # 如果提供了此参数，表示用户希望保存概念字典。
        # 如果需要保存概念字典，将命令行参数args.concept_dict_dir（概念字典的基本目录路径）
        # 与之前确定的suffix（根据条件生成的后缀）连接起来，形成完整的概念字典目录路径。
        concept_dict_dir = args.concept_dict_dir + suffix
        os.makedirs(concept_dict_dir, exist_ok=True)
        concept_dict_path = os.path.join(concept_dict_dir, f'{dataset}.json')

    print(f"Loading concepts from {concepts_json}")
    with open(concepts_json, 'r') as f:
        # 使用json.load函数从文件中读取概念字典，并将结果存储在concepts_dict变量中。
        concepts_dict = json.load(f)
    # 从概念字典中获取所有的键（即类名），并将它们转换为列表。
    gpt4_classes = list(concepts_dict.keys())
    # 根据指定的数据集（dataset），从CLASSES_DICT字典中获取对应的类名列表。
    tpt_classes = CLASSES_DICT[dataset]
    
    concept_embeds = {} # 创建一个空字典，用于存储生成的概念嵌入。
    concept_dict_all = {} # 创建一个空字典，用于存储与每个类别相关联的文本提示。
    # 为gpt4_classes中的每个类名生成一个包含其概念列表长度的数组。
    len_concepts = np.array([len(concepts_dict[classname_gpt4]) for classname_gpt4 in gpt4_classes])
    # 使用numpy的where函数找到长度为0的概念列表的索引。
    empty_indices = np.where(len_concepts == 0)[0]
    # 根据索引创建一个列表，包含所有没有概念的类名。
    empty_classnames = [gpt4_classes[i] for i in empty_indices]

    # 确保没有类名是空的。如果存在空的类名，将抛出异常，并显示空类名的列表。
    assert len(empty_classnames) == 0, f"Empty classnames: {empty_classnames}"
    # 遍历tpt_classes列表，将每个类别名称中的下划线（_）替换为空格。
    tpt_classes = [name.replace("_", " ") for name in tpt_classes]

    # 使用tqdm库来遍历tpt_classes和gpt4_classes，同时显示进度条。zip函数将两个列表中的元素配对。
    for classname_tpt, classname_gpt4 in tqdm(zip(tpt_classes, gpt4_classes), total=len(tpt_classes)):
        assert "_" not in classname_tpt

        # 从概念字典concepts_dict中获取与classname_gpt4对应的概念列表。
        concepts = concepts_dict[classname_gpt4]

        # 确保对于每个classname_gpt4，概念列表不为空
        assert len(concepts) > 0, f"Empty concepts for class {classname_gpt4} in dataset {dataset}"

        # 根据命令行参数args.no_cond或args.x_templates，以不同的方式生成文本提示：
        if args.no_cond: # 则直接使用概念作为提示。
            prompts = concepts
        elif args.x_templates:
            # 则使用ImageNet模板和make_descriptor_sentence函数生成更复杂的提示。
            prompts = [t.format(f"{classname_tpt}, " + make_descriptor_sentence(c)) for c in concepts for t in imagenet_templates]
        else:
            # 使用简单的格式"{classname_tpt}, {descriptor}"生成提示。
            prompts = [f"{classname_tpt}, " + make_descriptor_sentence(c) for c in concepts]
        # 对生成的文本提示进行标记化处理
        tokenized_prompts = tokenize(prompts).to(device)
        with torch.no_grad():
            # 使用CLIP模型的encode_text方法生成文本嵌入，并将结果移回CPU。
            embeds = clip.encode_text(tokenized_prompts).cpu()
        # 对生成的文本嵌入进行归一化处理。
        embeds = F.normalize(embeds, dim=-1)
        concept_embeds[classname_tpt] = embeds

        concept_dict_all[classname_tpt] = prompts
        
    
    if args.save_concept_dict:
        #concept_dict_path = os.path.join('ensemble/TPS', concept_dict_path) # 黄老添加
        print(f"Dumping concept dict to {concept_dict_path}")
        with open(concept_dict_path, 'w') as f:
            json.dump(concept_dict_all, f, indent=4)
            
    #concept_embeds_path = os.path.join('ensemble/TPS', concept_embeds_path) # 黄老添加
    print(f"Dumping concept embeds to {concept_embeds_path}")
    with open(concept_embeds_path, 'wb') as f:
        pickle.dump(concept_embeds, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 指定包含概念的JSON文件的路径，默认为 'concepts/imagenet-gpt4-full-v4.json'。
    parser.add_argument('--concepts_json', type=str, default='concepts/imagenet-gpt4-full-v4.json')
    parser.add_argument('--dataset', type=str, default='ImageNet')
    # --no_cond：如果提供此参数，则生成不包含特定条件的文本提示。
    parser.add_argument('--no_cond', action="store_true")
    # --x_templates：如果提供此参数，则使用ImageNet模板生成文本提示。
    parser.add_argument('--x_templates', action="store_true")
    parser.add_argument('--arch', type=str, default='ViT-B/16', choices=['ViT-B/16', 'RN50'])
    # --gpt4_concepts_embeds_dir：指定保存GPT-4生成的概念嵌入的目录，默认为 'concept_embeds_gpt4'。
    parser.add_argument('--gpt4_concepts_embeds_dir', type=str, default='concept_embeds_gpt4')
    # --concept_dict_dir：指定保存概念字典的目录，默认为 'concept_dict_gpt4'。
    parser.add_argument('--concept_dict_dir', type=str, default='concept_dict_gpt4')
    # --save_concept_dict：如果提供此参数，则保存概念字典。
    parser.add_argument('--save_concept_dict', action='store_true')

    args = parser.parse_args()

    save_concepts(args)
    