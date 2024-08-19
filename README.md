# TTA-Benchmark
课题组考核任务1
## 题干
背景:VLM(例如clip)具有较好的zero-shot泛化能力，然而在面对不同下游任务时(数据漂移)仍然存在性能下降的问题。需要提升clip-based model在目标域上的泛化能力，主要任务有图像分类、图像caption生成以及其余一些clip支持的任务。
任务一：选择一套现有的代码框架，将其扩展为VLM泛化任务的benchmark,需要包括zero-shot的一些经典方法(例如TPT,DiffTPT,TDA,SwapPrompt等)以及SOTA 方法，few-shot的一些经典方法(例如Tip-Adapter,CoOp等)以及SOTA方法，并给出采用相同模型结构下(resnet-base,vit-base)这些方法的性能对比，数据集采用(lmageNet-A,lmageNet-V2,ImageNet-R, andImageNet-S)和(Aircraft, Caltech101, Cars, DTD, EuroSAT, Flower102, Food101, Pets.SUN397, and UCF10)。
## Overviews

