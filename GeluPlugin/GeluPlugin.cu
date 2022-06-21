#include "GeluPlugin.cuh"
#include "common.cuh"
using namespace nvinfer1;

// constants for approximating the normal cdf
constexpr float A = 0.5f;
constexpr float B = 0.7978845608028654f;   // sqrt(2.0/M_PI)
constexpr float C = 0.035677408136300125f; // 0.044715 * sqrt(2.0/M_PI)

template<typename T>
__global__ void GeluKernel(const T a, const T b, const T c, int n, const T *input,T *output){
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index<n){
        const T in = input[index];
        const T cdf = a + a* tanh(in*(c*in*in+b));
        output[index] = in*cdf;
    }
}

GeluPlugin::GeluPlugin(const std::string name)
:mLayerName(name)
{
    WHERE_AM_I()
}

GeluPlugin::GeluPlugin(const std::string name, const void *data, size_t length):mLayerName(name) {
    WHERE_AM_I()
}

const char *GeluPlugin::getPluginVersion() const noexcept {
    WHERE_AM_I()
    return PLUGIN_VERSION;
}

const char *GeluPlugin::getPluginType() const noexcept {
    WHERE_AM_I()
    return PLUGIN_NAME;
}

const char *GeluPlugin::getPluginNamespace() const noexcept {
    WHERE_AM_I()
    return mNamespace.c_str();
}

int32_t GeluPlugin::getNbOutputs() const noexcept {
    WHERE_AM_I()
    return 1;
}

int32_t  GeluPlugin::initialize() noexcept {
    WHERE_AM_I()
    return 0;
}

void GeluPlugin::terminate() noexcept {
    WHERE_AM_I()
}

size_t GeluPlugin::getSerializationSize() const noexcept {
    WHERE_AM_I()
    // return sizeof(m_);
    return 0;
}

void GeluPlugin::serialize(void *buffer) const noexcept {
    WHERE_AM_I()
    // memcpy

}

void GeluPlugin::destroy() noexcept {
    WHERE_AM_I()

}

void GeluPlugin::setPluginNamespace(const char *pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

DataType GeluPlugin::getOutputDataType(int32_t index, const nvinfer1::DataType *inputTypes,
                                       int32_t nbInputs) const noexcept {
    WHERE_AM_I()
    return inputTypes[0];
}

void GeluPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas,
                                 IGpuAllocator *gpuAllocator) noexcept {
    WHERE_AM_I()
}

void GeluPlugin::detachFromContext() noexcept {
    WHERE_AM_I()
}

IPluginV2DynamicExt *GeluPlugin::clone() const noexcept {
    WHERE_AM_I()
    auto * p = new GeluPlugin(mLayerName);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

DimsExprs GeluPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs,
                                          IExprBuilder &exprBuilder) noexcept {
    WHERE_AM_I()
    return inputs[0];
}

bool GeluPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs,
                                           int32_t nbOutputs) noexcept {
    WHERE_AM_I()
    switch(pos){
        case 0:
            return(inOut[0].type==DataType::kFLOAT || inOut[0].type==DataType::kHALF) && (inOut[0].format==TensorFormat::kLINEAR);
        case 1:
            return (inOut[1].type == inOut[0].type) && (inOut[1].format == inOut[0].format);
        default:
            return false;
    }
    return false;
}

void GeluPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs,
                                 const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept {
    WHERE_AM_I()
}

size_t GeluPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs,
                                    int32_t nbOutputs) const noexcept {
    WHERE_AM_I()
    return 0;
}


int32_t GeluPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc,
                            const void *const *inputs, void *const *outputs, void *workspace,
                            cudaStream_t stream) noexcept {
    WHERE_AM_I()
    const int inputVolume = volume(inputDesc[0].dims);
    //block和grid尺寸
    constexpr int blocksize = 256;
    const int gridsize = (inputVolume + blocksize - 1) / blocksize;

    if(inputDesc[0].type==DataType::kFLOAT){
        const float* input = static_cast<const float*>(inputs[0]);
        float* output = static_cast<float*>(outputs[0]);

        (GeluKernel<float>)<<<gridsize,blocksize,0,stream>>>(A,B,C,inputVolume,input,output);
    }
    else if (inputDesc[0].type==DataType::kHALF){
        if(0==(inputVolume&1)){
            const int n2 = inputVolume/2;

            const int gridsize2 = (n2 + blocksize - 1)/blocksize;
            const half2 A2 = __floats2half2_rn(A,A);
            const half2 B2 = __floats2half2_rn(B,B);
            const half2 C2 = __floats2half2_rn(C,C);
            const half2* input2 = reinterpret_cast<const half2*>(inputs[0]);
            half2* output2 = reinterpret_cast<half2*>(outputs[0]);
            GeluKernel<half2><<<gridsize,blocksize,0,stream>>>(A2,B2,C2,n2,input2,output2);
        }
        else{
            const half * input = static_cast<const half*>(inputs[0]);
            half * output = static_cast<half*>(outputs[0]);
        
            GeluKernel<__half><<<gridsize,blocksize,0,stream>>>(A,B,C,inputVolume,input,output);
        }
    }
    else{
        printf("Unsupport datatype!\n");
    }
    return 0;
}

PluginFieldCollection GeluPluginCreator::fc_ {};
std::vector<PluginField> GeluPluginCreator::attr_;

GeluPluginCreator::GeluPluginCreator() {
    WHERE_AM_I()
    attr_.clear();

    fc_.nbFields = attr_.size();
    fc_.fields = attr_.data();
}

GeluPluginCreator::~GeluPluginCreator() noexcept {
    WHERE_AM_I();
}

IPluginV2 *GeluPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept {
    WHERE_AM_I()
    return new GeluPlugin(name);
}

IPluginV2 *GeluPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept {
    WHERE_AM_I()
    return new GeluPlugin(name,serialData,serialLength);
}

void GeluPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept {
    WHERE_AM_I()
    namespace_ = pluginNamespace;
}
const char *GeluPluginCreator::getPluginNamespace() const noexcept {
    WHERE_AM_I()
    return namespace_.c_str();
}

const char *GeluPluginCreator::getPluginName() const noexcept {
    WHERE_AM_I()
    return PLUGIN_NAME;
}

const char *GeluPluginCreator::getPluginVersion() const noexcept {
    WHERE_AM_I()
    return PLUGIN_VERSION;
}

const PluginFieldCollection *GeluPluginCreator::getFieldNames() noexcept {
    WHERE_AM_I()
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(GeluPluginCreator);