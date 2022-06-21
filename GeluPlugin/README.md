# Gelu 激活函数
## intro
在NLP和语言任务中较为常用，当前有许多Transformer也在使用这个激活函数，PyTorch的实现为torch.nn.Relu() \
![截屏2022-06-21 20 22 19](https://user-images.githubusercontent.com/71363087/174797916-dcfbe9bc-cd85-4340-aae1-f377d122bcbf.png) \
具体的公式为：\
![截屏2022-06-21 20 30 48](https://user-images.githubusercontent.com/71363087/174799458-b4ae9620-f500-4962-8dfa-0d5c0763e75a.png) \
将高斯分布展开，得到最终的计算公式为：\
![截屏2022-06-21 20 31 37](https://user-images.githubusercontent.com/71363087/174799615-2a7deccd-bef4-4f9a-8b03-30ee9a629fb4.png)
## 输入输出
支持FP32和FP16输入 \
支持动态尺寸推理,相比原始onnx转出的engine，显存使用量从 3G->0.8G，在batchsize16尺度下推理速度 0.17->0.10 

<img width="788" alt="截屏2022-06-21 20 34 57" src="https://user-images.githubusercontent.com/71363087/174800253-d7cbd852-c48d-4a59-96c1-c36c97d738f3.png">
## 注意事项
由于增加了对FP16的支持，tanh函数重载过，编译时需要包含common.h文件（Nvidia官方版本）\
编译时注意Makefile显卡运算能力SM的设置，对于Nvidia-A10，运算能力为8.0，SM=80 \
具体的显卡算力列表见：\
https://developer.nvidia.com/cuda-gpus
