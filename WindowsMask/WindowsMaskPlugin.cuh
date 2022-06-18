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

#include <vector>
#include <string>
#include <map>
#include <NvInfer.h>
#include <cuda_fp16.h>

// +------- Debug wrapper --------------------------------------------------------------------------

#define WHERE_AM_I() do {printf("[%s]: this=->%p\n",__func__,this);} while(0);



#define CEIL_DIVIDE(X, Y) (((X) + (Y)-1) / (Y))
#define CEIL_TO(X, Y)     (CEIL_DIVIDE(X, Y) * (Y))

// +------- Plguin ---------------------------------------------------------------------------------
namespace
{
    static const char* PLUGIN_NAME{"WindowsMask"};
    static const char* PLUGIN_VERSION{"1"};
} // namespace

namespace nvinfer1
{
// +------- Plugin body ----------------------------------------------------------------------------
    class WindowsMaskPlugin: public IPluginV2DynamicExt
    {
    private:
        const std::string name_;
        std::string namespace_;
        struct {
            int window_size;
            int shift_size;
        }m_;


    public:
        WindowsMaskPlugin() = delete;
        
        WindowsMaskPlugin(const std::string& name,int win_size, int s_size) : name_(name)
        {
                            printf("111111111");
            WHERE_AM_I();

            m_.window_size = win_size;
            m_.shift_size = s_size;
        }

        WindowsMaskPlugin(const std::string& name, const void* data, size_t length) : name_(name)
        {
                        printf("data length:%d\n",length);
                        printf("need length:%d\n",sizeof(m_));

                        printf("2222222222");
            WHERE_AM_I();

            memcpy(&m_,data,sizeof(m_));
        }


        ~WindowsMaskPlugin()
        {
            WHERE_AM_I();
        }

        size_t getSerializationSize() const noexcept override
                {
                        WHERE_AM_I();
                return sizeof(m_);
                }

        void serialize(void *buffer) const noexcept override
                {
                        WHERE_AM_I();
                        memcpy(buffer,&m_,sizeof(m_));
                }

        IPluginV2DynamicExt* clone() const noexcept override
                {
                        WHERE_AM_I();
                        auto p = new WindowsMaskPlugin(name_,&m_,sizeof(m_));
                        p->setPluginNamespace(namespace_.c_str());
                        return p;
                }

        int getNbOutputs() const noexcept override
                {
                        WHERE_AM_I();
                return 1;
                }

        DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override
                {
                        WHERE_AM_I();
                return inputs[0];
                }

        bool supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
                {
                        WHERE_AM_I();
                if(inOut[pos].format != TensorFormat::kLINEAR)
                {
                    return false;
                }
                //输入0是mask，输入1是shape，输出为2
                bool res = false;
                switch(pos)
                {
                    case 0:
                        res = (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF); break;  //可以是FP32或者FP16
                    case 1:
                        res = (inOut[pos].type == DataType::kINT32); break;
                    case 2:
                        res = (inOut[2].type == inOut[0].type); break; //输入和输出的数据类型相符合即可
                    default:// should NOT be here
                        res = false;
                }
                return res;
                }

        DataType getOutputDataType(int outputIndex, const DataType* inputTypes, int nbInputs) const noexcept override
                {
                        WHERE_AM_I();
                        return inputTypes[0]; //要想支持FP16的话，需要与输入的格式相同吧

                }

        void configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs,const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override
                {
                        WHERE_AM_I();
                }

        size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs,int32_t nbOutputs) const noexcept override
                {
                        WHERE_AM_I();
                return 0;
                }

        void setPluginNamespace(const char* szNamespace) noexcept override
                {
                        WHERE_AM_I();
                namespace_ = szNamespace;
                }
        const char* getPluginNamespace() const noexcept override
                {
                        WHERE_AM_I();
                return namespace_.c_str();
                }
        const char* getPluginType() const noexcept override
                {
                        WHERE_AM_I();
                return PLUGIN_NAME;
                }
        const char* getPluginVersion() const noexcept override
                {
                        WHERE_AM_I();
                return PLUGIN_VERSION;
                }
        int32_t initialize() noexcept override
                {
                        WHERE_AM_I();
                return 0;
                }
        void terminate() noexcept override
                {
                        WHERE_AM_I();
                return;
                }

        void destroy() noexcept override
                {
                        WHERE_AM_I();
                }
        void attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas,
                                            IGpuAllocator *gpuAllocator) noexcept {
                WHERE_AM_I()                        
                }
        
        void detachFromContext() noexcept {
                WHERE_AM_I()
                }

        int32_t enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    }; // class WindowsMaskPlugin

    class WindowsMaskPluginCreator : public IPluginCreator
    {
    private:
        static PluginFieldCollection fc_;
        static std::vector<PluginField> attr_;
        std::string namespace_;

    public:
        WindowsMaskPluginCreator()
        {
                WHERE_AM_I();
            fc_.nbFields = attr_.size();
            fc_.fields = attr_.data();
        }

        ~WindowsMaskPluginCreator() {}

        IPluginV2 *createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override
                {
                        WHERE_AM_I();
                        int win_size = 8;
                        int s_size = 4;
                        std::map<std::string,int *>parameterMap{
                                {"window_size",&win_size},
                                {"shift_size",&s_size}
                        };
                        for (int i = 0; i < fc->nbFields; ++i)
                        {
                                printf("nbfield%d\n",i);
                            if (parameterMap.find(fc->fields[i].name) != parameterMap.end())
                            {
                                printf("i got it !");
                                *parameterMap[fc->fields[i].name] = *reinterpret_cast<const int *>(fc->fields[i].data);
                            }
                        }
                        return new WindowsMaskPlugin(name,win_size,s_size);
                }

        IPluginV2 *deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override
                {
                        WHERE_AM_I();
                        return new WindowsMaskPlugin(name, serialData, serialLength);
                }

        void setPluginNamespace(const char* szNamespace) noexcept override
                {
                        WHERE_AM_I();
                        namespace_ = szNamespace;
                }

        const char* getPluginNamespace() const noexcept override
                {
                        WHERE_AM_I();
                        return namespace_.c_str();
                }

        const char* getPluginName() const noexcept override
                {
                        WHERE_AM_I();
                        return PLUGIN_NAME;
                }

        const char* getPluginVersion() const noexcept override
                {
                        WHERE_AM_I();
                        return PLUGIN_VERSION;
                }

        const PluginFieldCollection* getFieldNames() noexcept override
                {
                        WHERE_AM_I();
                        return &fc_;
                }
    }; // class WindowsMaskPluginCreator

    REGISTER_TENSORRT_PLUGIN(WindowsMaskPluginCreator);
} // namespace nvinfer1

