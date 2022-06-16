from re import X
import torch
import torch.nn as nn
import onnx
import onnx_graphsurgeon as gs
import numpy as np

from collections import OrderedDict
#先定义一个基本的线性变换网络
class AffineModel(nn.Module):
    def __init__(self,k,b) -> None:
        super().__init__()
        self.k = k
        self.b = b

    def forward(self,x:torch.Tensor)->torch.Tensor:

        return self.k*x+self.b
#测试模型效果
model = AffineModel(2,1).cuda()
x = np.ones((1,256,256))
x = torch.from_numpy(x).cuda()
y = model(x)
print(y)
# 操作可行，导出onnx
x = torch.randn((1,256,256),requires_grad=False).cuda()
onnx_file = 'affine.onnx'
torch.onnx.export(
    model,
    (x),
    onnx_file,
    verbose=False,
    do_constant_folding=True,
    opset_version=13,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={ # 动态维度为batch
        "input":{0:'batch'},
        "output":{0:"batch"}
    }
)
# 图修改，替换原始节点
onnx_model = onnx.load(onnx_file)
graph = gs.import_onnx(onnx_model)
inputs = graph.inputs
outputs = graph.outputs

new_node = gs.Node(op='AffineTrans',name='AffineTrans-1',\
    attrs=OrderedDict([
        ['k',2.0],
        ['b',1.0]
    ]))

new_node.inputs = inputs
new_node.outputs = outputs
for node in graph.nodes:
    node.outputs.clear()
graph.nodes.append(new_node)

graph.cleanup()
onnx.save(gs.export_onnx(graph),'affine_surgeon.onnx')

    

