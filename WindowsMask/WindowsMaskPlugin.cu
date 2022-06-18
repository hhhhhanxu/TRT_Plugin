/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "WindowsMaskPlugin.cuh"

using namespace nvinfer1;

PluginFieldCollection WindowsMaskPluginCreator::fc_{};
std::vector<PluginField> WindowsMaskPluginCreator::attr_;

template<typename T>
__global__ void windowsMaskKernel(const T *pInput, const int *shape, T *pOutput, int nElement, int window_size, int shift_size)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index > nElement){
        return;
    }
    if (shift_size == 0){
        pOutput[index] = 8;
        return ;
    }


    int H = shape[1];
    int W = shape[2];

    int h = index / W;
    int w = index % W;

    int h_index = 0;
    int w_index = 0;

    if(h >= (H-window_size)){
        if( h >= (H-shift_size))
            h_index = 2;
        else
            h_index = 1;
    }
    if(w >= (W-window_size)){
        if( w >= (W-shift_size))
            w_index = 2;
        else
            w_index = 1;
    }
    //是否需要强制类型转换为T呢
    pOutput[index] = h_index * 3 + w_index;

}

template __global__ void windowsMaskKernel<float> (const float * pInput, const int * shape,float * pOutput, int nElement, int window_size, int shift_size);
template __global__ void windowsMaskKernel<__half> (const __half * pInput, const int * shape,__half * pOutput, int nElement, int window_size, int shift_size);

int32_t WindowsMaskPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I()
    int nElement = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; i++)
    {
        nElement *= inputDesc[0].dims.d[i];
    }
    //每个block64线程 -> 32线程
    dim3 grid(CEIL_DIVIDE(nElement, 32), 1, 1), block(32, 1, 1);
    //输入0应该是mask
    if(inputDesc[0].type==DataType::kFLOAT){
        const auto input = static_cast<const float *>(inputs[0]);
        const auto shape = static_cast<const int *>(inputs[1]);
        auto output = static_cast<float *>(outputs[0]);

        (windowsMaskKernel<float>)<<<grid,block,0,stream>>>(input,shape,output,nElement,m_.window_size,m_.shift_size);
    }
    else if(inputDesc[0].type==DataType::kHALF){
        const auto input = static_cast<const half *>(inputs[0]);
        const auto shape = static_cast<const int *>(inputs[1]);
        auto output = static_cast<half *>(outputs[0]);

        (windowsMaskKernel<__half>)<<<grid,block,0,stream>>>(input,shape,output,nElement,m_.window_size,m_.shift_size);
    }
    else{
        printf("Unsupport datatype!\n");
    }
    // windowsMaskKernel <<<grid, block, 0, stream>>>((float *)inputs[0], (int *)inputs[1], (float *)outputs[0], nElement);
    return 0;
}



REGISTER_TENSORRT_PLUGIN(WindowsMaskPluginCreator);

