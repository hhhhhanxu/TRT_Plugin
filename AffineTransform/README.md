# 仿射变换
一个非常简单的plugin算子，主要也是为了练手用的 \
对于输入x \
输出y=kx+b 
## 输入输出
支持FP32和FP16 \
支持动态尺寸推理 
## 注意事项
编译时注意Makefile显卡运算能力SM的设置，对于Nvidia-A10，运算能力为8.0，SM=80 \
具体的显卡算力列表见：\
https://developer.nvidia.com/cuda-gpus

