# Swin Transformer-Shift Window Mask变换
## Intro
在Swin Transform中，shift window是一个很精妙的设计，但是移位（torch.roll）操作带来的边界问题会导致不能直接计算attention，需要一个Mask来为空间上不连续的部分做一个掩码. \
在实际代码中，基于window size和shift size对mask进行切分，并为mask的不同区域分配id，用于后续操作实现window级的mask \
![mask_cal](https://user-images.githubusercontent.com/71363087/174447765-98cbc59e-89b5-475c-a130-e0f89ebe0dc4.jpg)
本plugin实现的就是上述的操作 \
输入参数有2个，input为与mask同维度的Tensor，shape具体为mask的shape \
输出如上图所示，是一个形状与input一样（仅指Width和Height）的Tensor 

## 输入输出
支持FP32和FP16 \
支持动态尺寸推理 \
由于原始代码中有3次slice和对应的设置mask值操作，直接转出会生成9组同类的节点组，大概共计700-800个节点，转TRT对显存要求较高 \
<img width="525" alt="截屏2022-06-19 00 34 07" src="https://user-images.githubusercontent.com/71363087/174448168-ec49e710-148b-4460-b47f-7b4f9f7816cc.png"> \
（⬆️原始mask op 计算图的一组节点，一共9组⬆️）\
计算图结构变化：\
<img width="1073" alt="截屏2022-06-19 00 41 48" src="https://user-images.githubusercontent.com/71363087/174448351-9f066b93-9ac3-4f87-b3ee-154aabce3287.png"> 

使用plugin之后，完全替代该部分节点，并且在速度上仅为直接转出的1/4 

## 注意事项
编译时注意Makefile显卡运算能力SM的设置，对于Nvidia-A10，运算能力为8.0，SM=80 \
具体的显卡算力列表见：\
https://developer.nvidia.com/cuda-gpus
