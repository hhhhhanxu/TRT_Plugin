import onnx
import onnx_graphsurgeon as gs

import os
import sys
import ctypes
import numpy as np
from glob import glob 
from time import time_ns
from datetime import datetime as dt
from cuda import cudart
import tensorrt as trt
from collections import OrderedDict
import cv2


plan_file = "affine.plan"
plugin_path = "./"
soFileList = glob(plugin_path + "*.so")

logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')

if len(soFileList) > 0:
    print("Find Plugin %s!"%soFileList)
else:
    print("No Plugin!")
for soFile in soFileList:
    ctypes.cdll.LoadLibrary(soFile)
#-------------------------------------------------------------------------------

print("Test Plan!")
if os.path.isfile(plan_file):
    with open(plan_file, 'rb') as encoderF:
        engine = trt.Runtime(logger).deserialize_cuda_engine(encoderF.read())
    if engine is None:
        print("Failed loading %s"%plan_file)
        exit()
    print("Succeeded loading %s"%plan_file)
else:
    print("Failed finding %s"%plan_file)
    exit()
nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
nOutput = engine.num_bindings - nInput
context = engine.create_execution_context()
#-------------------------------------------------------------------------------
test_data = np.zeros((1,256,256),dtype=np.float32)

context.set_binding_shape(0, test_data.shape)

bufferH = []
bufferH.append( test_data.reshape(-1) )

for i in range(nInput, nInput + nOutput):                
    bufferH.append( np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))) )

bufferD = []
for i in range(nInput + nOutput):                
    bufferD.append( cudart.cudaMalloc(bufferH[i].nbytes)[1] )

for i in range(nInput):
    cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

context.execute_v2(bufferD)

for i in range(nInput, nInput + nOutput):
    cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

# warm up
for i in range(10):
    context.execute_v2(bufferD)

# test infernece time
t0 = time_ns()
for i in range(30):
    context.execute_v2(bufferD)
t1 = time_ns()
timePerInference = (t1-t0)/1000/1000/30

index_output = engine.get_binding_index("output")
output = bufferH[index_output]

print(output)
print(timePerInference)
# save image
# output = np.clip(np.squeeze(output), 0.0, 1.0)
# if output.ndim == 3:
#     output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
# output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

