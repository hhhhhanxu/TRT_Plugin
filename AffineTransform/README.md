# 仿射变换
一个非常简单的plugin算子，主要也是为了练手用的 \
对于输入x \
输出y=kx+b \
<img width="462" alt="截屏2022-06-16 14 49 50" src="https://user-images.githubusercontent.com/71363087/174009382-cb59418d-853c-4e8a-b03d-4091fc63b3dc.png">

## 输入输出
支持FP32和FP16 \
支持动态尺寸推理 \
<img width="986" alt="截屏2022-06-16 14 57 04" src="https://user-images.githubusercontent.com/71363087/174010710-6cdeabab-85de-4e0d-92b7-7d62330fc1ce.png">

## 注意事项
编译时注意Makefile显卡运算能力SM的设置，对于Nvidia-A10，运算能力为8.0，SM=80 \
具体的显卡算力列表见：\
https://developer.nvidia.com/cuda-gpus

