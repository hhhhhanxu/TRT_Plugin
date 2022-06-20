#include "WindowsFillPlugin.cuh"

using namespace nvinfer1;

template<typename T>
__global__ void fillKernel(const T *pInput,T *pOutput){
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(pInput[index] != T(0)){
        pOutput[index] = T(-100.0);
    }
}

template __global__ void fillKernel<float>(const float * pInput,float *pOutput);
template __global__ void fillKernel<__half>(const __half * pInput,__half *pOutput);


FillPlugin::FillPlugin(const std::string &name):name_(name) {
    WHERE_AM_I()
}

FillPlugin::FillPlugin(const std::string &name, const void *buffer, size_t length):name_(name) {
    WHERE_AM_I()
}

FillPlugin::~FillPlugin() noexcept {
    WHERE_AM_I()
}

const char *FillPlugin::getPluginType() const noexcept {
    WHERE_AM_I()
    return PLUGIN_NAME;
}

const char *FillPlugin::getPluginVersion() const noexcept {
    WHERE_AM_I()
    return PLUGIN_VERSION;
}

int32_t FillPlugin::getNbOutputs() const noexcept {
    WHERE_AM_I()
    return 1;
}

int32_t FillPlugin::initialize() noexcept {
    WHERE_AM_I()
    return 0;
}

void FillPlugin::terminate() noexcept {
    WHERE_AM_I()
    return ;
}

size_t FillPlugin::getSerializationSize() const noexcept {
    WHERE_AM_I()
    //如果plugin内带参数m_的话，这里应当是返回sizeof(m_)
    return 0;
}

void FillPlugin::serialize(void *buffer) const noexcept {
    WHERE_AM_I()
}

void FillPlugin::destroy() noexcept {
    WHERE_AM_I()
}

void FillPlugin::setPluginNamespace(const char *pluginNamespace) noexcept {
    WHERE_AM_I()
    namespace_ = pluginNamespace;
}

const char *FillPlugin::getPluginNamespace() const noexcept {
    WHERE_AM_I()
    return namespace_.c_str();
}

DataType FillPlugin::getOutputDataType(int32_t index, const nvinfer1::DataType *inputTypes,
                                       int32_t nbInputs) const noexcept {
    //直接硬返回FLOAT？
    //return DataType::kFLOAT;
    return inputTypes[0];
}

void FillPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas,
                                 IGpuAllocator *gpuAllocator) noexcept {
    WHERE_AM_I()
}

void FillPlugin::detachFromContext() noexcept {
    WHERE_AM_I()
}

IPluginV2DynamicExt *FillPlugin::clone() const noexcept {
    WHERE_AM_I()
    return new FillPlugin(name_);
}

DimsExprs FillPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs,
                                          IExprBuilder &exprBuilder) noexcept {
    WHERE_AM_I()
    return inputs[0];
}

bool FillPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs,
                                           int32_t nbOutputs) noexcept {
    WHERE_AM_I()
    if(inOut[pos].format != TensorFormat::kLINEAR)
    {
        return false;
    }

    bool res = false;
    switch (pos) {
        case 0:
            res = (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF); break;
        case 1:
            res = (inOut[pos].type == inOut[0].type);break;
        default:
            res = false;
    }
    return res;
}

void FillPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs,
                                 const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept {
    WHERE_AM_I()
}

size_t FillPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs,
                                    const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept {
    WHERE_AM_I()
    return 0;
}

int32_t FillPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc,
                            const void *const *inputs, void *const *outputs, void *workspace,
                            cudaStream_t stream) noexcept {
    int nElement = 1;
    for(int i =0;i<inputDesc[0].dims.nbDims;i++){
        nElement *= inputDesc[0].dims.d[i];
    }
    //thread size 32
    dim3 grid(CEIL_DIVIDE(nElement,32),1,1),block(32,1,1);
    //启动kernel
    if(inputDesc[0].type==DataType::kFLOAT){
        const auto input = static_cast<const float *>(inputs[0]);
        auto output = static_cast<float *>(outputs[0]);

        (fillKernel<float>)<<<grid,block,0,stream>>>(input,output);
    }
    else if(inputDesc[0].type==DataType::kHALF){
        const auto input = static_cast<const half *>(inputs[0]);
        auto output = static_cast<half *>(outputs[0]);
        printf("Using FP16 kernel !!!");
        (fillKernel<__half>)<<<grid,block,0,stream>>>(input,output);
    }
    else{
        printf("Unsupport datatype!\n");
    }
    return 0;
}


PluginFieldCollection FillPluginCreator::fc_{};
std::vector<PluginField> FillPluginCreator::attr_;

FillPluginCreator::FillPluginCreator() {
    attr_.clear();
    WHERE_AM_I()
    fc_.nbFields = attr_.size();
    fc_.fields = attr_.data();
}

FillPluginCreator::~FillPluginCreator() noexcept {
    WHERE_AM_I()
}

IPluginV2 *FillPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept {
    WHERE_AM_I()
    return new FillPlugin(name);
}

IPluginV2 *FillPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept {
    WHERE_AM_I()
    return new FillPlugin(name,serialData,serialLength);
}

void FillPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept {
    WHERE_AM_I()
    namespace_ = pluginNamespace;
}

const char *FillPluginCreator::getPluginNamespace() const noexcept {
    WHERE_AM_I()
    return namespace_.c_str();
}

const char *FillPluginCreator::getPluginName() const noexcept {
    WHERE_AM_I()
    return PLUGIN_NAME;
}

const char *FillPluginCreator::getPluginVersion() const noexcept {
    WHERE_AM_I()
    return PLUGIN_VERSION;
}

const PluginFieldCollection *FillPluginCreator::getFieldNames() noexcept {
    WHERE_AM_I()
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(FillPluginCreator);
