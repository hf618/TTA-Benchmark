import argparse
import os
import torch
import torch.nn.functional as F
import pickle

from tqdm import tqdm

import sys
sys.path.append(os.getcwd())

from ensemble.TPS.model import load, tokenize, DOWNLOAD_ROOT
from ensemble.TPS.model.text_encoders import TextEncoderWithPrompt
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
    "flower": flower102_classes,
    "food101": food101_classes,
    "food": food101_classes,
    "UCF101": ucf101_classes,
    "pets": pets_classes,
    "EuroSAT": eurosat_classes,
    "ImageNet": imagenet_classes,
    "SUN397": sun397_classes,
    "CalTech101": caltech101_classes,
    "cars": cars_classes,
}

device='cuda'

n_ctx = 4
#
coop_path = '/root/autodl-tmp/pretrained/to_gdrive/vit_b16_ep50_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50'
ctx = torch.load(coop_path)['state_dict']['ctx'].unsqueeze(0)


def main(args):

    clip, _, _ = load(args.arch, device=device, download_root=DOWNLOAD_ROOT)
    # 从加载的CLIP模型的视觉部分的第一个卷积层的权重中获取数据类型。
    # 这通常用于确保数据类型的一致性，特别是在进行类型转换或确保模型输入输出具有相同数据类型时。
    dtype = clip.visual.conv1.weight.dtype
    # 使用加载的CLIP模型实例创建一个TextEncoderWithPrompt对象。
    # 这个对象可能是用来扩展或修改CLIP的文本编码器，使其能够处理提示（prompts），这在生成文本嵌入时非常有用。
    text_encoder_w_prompt = TextEncoderWithPrompt(clip)

    if args.x_templates:
        # 如果此属性为True，则使用'class_embeds_w_imagenet_templates'作为目录名称。
        dir_class_embeds = 'class_embeds_w_imagenet_templates'
    elif args.coop:
        dir_class_embeds = 'coop_embeds'
    else:
        dir_class_embeds = 'class_embeds'
    # 使用os.path.join函数将基础目录与条件判断得到的目录名称拼接起来，形成完整的目录路径。
    dir_class_embeds = os.path.join(args.arch.replace('/', '-').lower() + "_embeds", dir_class_embeds)

    os.makedirs(dir_class_embeds, exist_ok=True)

    print(f"Saving class embeds to {dir_class_embeds}")

    DATASETS = ["EuroSAT", "aircraft", "DTD", "flower", "food101", "UCF101", "SUN397", "CalTech101", "cars", "pets", "ImageNet"]

    for dataset in DATASETS:
        print(f"Dataset: {dataset}")
        #root = 'ensemble/TPS' # 确保在正常的目录
        #class_embeds_path = os.path.join(root, dir_class_embeds, f'{dataset}.pkl')
        class_embeds_path = os.path.join(dir_class_embeds, f'{dataset}.pkl')

        classes_lst = CLASSES_DICT[dataset]
        classes_lst = [name.replace("_", " ") for name in classes_lst]

        class_embeds = {}
        for classname in tqdm(classes_lst, total=len(classes_lst)):
            assert "_" not in classname

            # 根据args.x_templates和args.coop的值，为每个类别生成相应的文本提示（prompts）。
            if args.x_templates:
                prompts = [template.format(classname) for template in imagenet_templates]
            else:
                prompts = [f'a photo of a {classname}.']
            # 使用tokenize函数将文本提示转换为模型能理解的标记序列，并将这些标记移动到设定的设备上。
            tokenized_prompts = tokenize(prompts).to(device)
            if args.coop:
                with torch.no_grad():
                    # 使用clip.token_embedding获取标记的初始嵌入。
                    embedding = clip.token_embedding(tokenized_prompts).type(dtype)
                # 构造包含特定前缀、上下文和后缀的提示张量。
                prefix = embedding[:, :1, :]
                suffix = embedding[:, 1 + n_ctx :, :]  # CLS, EOS

                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
                # 使用自定义的text_encoder_w_prompt生成嵌入。
                embeds = text_encoder_w_prompt(prompts, tokenized_prompts)
            # 如果不使用CoOp，直接使用CLIP模型的encode_text方法生成嵌入。
            else:
                with torch.no_grad():
                    embeds = clip.encode_text(tokenized_prompts).cpu()
            # 使用F.normalize对嵌入向量进行归一化处理。
            embeds = F.normalize(embeds, dim=-1)
            # 将归一化的嵌入向量存储到class_embeds字典中，键是类别名称，值是对应的嵌入向量。
            class_embeds[classname] = embeds.squeeze()
        # 使用pickle.dump将class_embeds字典序列化并保存到.pkl文件中。
        print(f"Dumping class embeds to {class_embeds_path}")
        with open(class_embeds_path, 'wb') as f:
            pickle.dump(class_embeds, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='ViT-B/16', choices=['ViT-B/16', 'RN50'])
    # 定义一个可选参数x_templates，使用action='store_true'表示如果该参数被包含在命令行中，则对应的变量将被设置为True。
    parser.add_argument('--x_templates', action='store_true', help='whether to use imagenet templates')
    # 类似于x_templates，定义了一个可选参数coop，用于指示是否使用CoOp前缀。
    parser.add_argument('--coop', action='store_true', help='whether to use coop prefix')

    args = parser.parse_args()

    main(args)
