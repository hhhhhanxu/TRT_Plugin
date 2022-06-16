#include "AffineTrans.cuh"

//实现仿射变换对kernel
template<typename T>
__global__ void AffineTransKernel(const T * input, T * output ,const T k,const T b,const int nElement){
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index>=nElement)
        return ;

    T _1 = input[index];
    T _2 = k * _1 +b;

    output[index] = _2;
}
namespace nvinfer1{
    AffineTransPlugin::AffineTransPlugin(const std::string &name, float k, float b):name_(name) {
        //因为name_是const类型，所以必须用初始化列表
        //每个函数都调用一次就完了
        WHERE_AM_I()
        m_.k = k;
        m_.b = b;
    }

    AffineTransPlugin::AffineTransPlugin(const std::string &name, const void *buffer, size_t length):name_(name) {
        WHERE_AM_I()
        // printf("Affine create ing !!!!!!!!!! \n");
        memcpy(&m_,buffer,sizeof(m_));
    }

    AffineTransPlugin::~AffineTransPlugin() noexcept {
        WHERE_AM_I()
    }

    IPluginV2DynamicExt *AffineTransPlugin::clone() const noexcept {
        WHERE_AM_I()
        auto p = new AffineTransPlugin(name_,&m_,sizeof(m_));
        p->setPluginNamespace(namespace_.c_str());
        return p;
    }

    //这个函数干嘛的
    int32_t AffineTransPlugin::getNbOutputs() const noexcept {
        WHERE_AM_I()
        return 1;
    }

    //获取输出的数据类型，一般与输入相同
    DataType AffineTransPlugin::getOutputDataType(int32_t index, const nvinfer1::DataType *inputTypes,
                                                  int32_t nbInputs) const noexcept{
        WHERE_AM_I()
        return inputTypes[0];
    }

    DimsExprs AffineTransPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs,
                                                    IExprBuilder &exprBuilder) noexcept {
        //重点函数,用于确定输出维度
        WHERE_AM_I()
        return inputs[0];
    }

    bool AffineTransPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs,
                                                      int32_t nbOutputs) noexcept {
        //下面的case只写0和1说明是单输入单输出
        WHERE_AM_I()
        switch(pos){
            case 0:
                return (inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF) && inOut[0].format == TensorFormat::kLINEAR;  //输入需要满足的参数格式
            case 1:
                return inOut[1].type == inOut[0].type && inOut[1].format == inOut[0].format;  //输入和输出必须相符合
            default:
                return false;
        }
        return false;
    }

    void AffineTransPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs,
                                            const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept {
        WHERE_AM_I()
    }

    size_t AffineTransPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs,
                                             const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept {
        WHERE_AM_I()
        return 0;
    }

    int32_t AffineTransPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc,
                                       const void *const *inputs, void *const *outputs, void *workspace,
                                       cudaStream_t stream) noexcept {
        WHERE_AM_I()
        int nElement = 1;
        for(int i=0; i<inputDesc[0].dims.nbDims;i++){
            nElement *= inputDesc[0].dims.d[i];
        }//计算输入Tensor尺寸？
        dim3 block_size(CEIL_DIVIDE(nElement,256),1,1);
        dim3 thread_size(256,1,1);
        if(inputDesc[0].type == DataType::kFLOAT){
            printf("FP32 kernel!\n");
            (AffineTransKernel<float>)<<<block_size,thread_size,0,stream>>>(reinterpret_cast<const float *>(inputs[0]),reinterpret_cast<float *>(outputs[0]),m_.k,m_.b,nElement);
        }
        else if(inputDesc[0].type == DataType::kHALF){
            printf("FP16 kernel!\n");
            //由于k和b初始化都是默认float的，所以需要转一下half
            (AffineTransKernel<__half>)<<<block_size,thread_size,0,stream>>>(reinterpret_cast<const __half *>(inputs[0]),reinterpret_cast<__half *>(outputs[0]),__half(m_.k),__half(m_.b),nElement);
        }
        else{
            printf("Unsupport datatype!\n");
        }
        return 0;
    }

    void AffineTransPlugin::destroy() noexcept {
        WHERE_AM_I()
    }

    int32_t AffineTransPlugin::initialize() noexcept {
        WHERE_AM_I()
        return 0;
    }

    void AffineTransPlugin::terminate() noexcept {
        WHERE_AM_I()
    }

    size_t AffineTransPlugin::getSerializationSize() const noexcept {
        WHERE_AM_I()
        return sizeof(m_);
    }

    void AffineTransPlugin::serialize(void *buffer) const noexcept {
        WHERE_AM_I()
        memcpy(buffer,&m_,sizeof(m_));
    }

    void AffineTransPlugin::setPluginNamespace(const char *pluginNamespace) noexcept {
        WHERE_AM_I()
        namespace_ = pluginNamespace;
    }

    const char *AffineTransPlugin::getPluginNamespace() const noexcept {
        WHERE_AM_I()
        return namespace_.c_str();
    }

    const char *AffineTransPlugin::getPluginType() const noexcept {
        WHERE_AM_I()
        return PLUGIN_NAME;
    }

    const char *AffineTransPlugin::getPluginVersion() const noexcept {
        WHERE_AM_I()
        return PLUGIN_VERSION;
    }

    void AffineTransPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas,
                                            IGpuAllocator *gpuAllocator) noexcept {
        WHERE_AM_I()
    }

    void AffineTransPlugin::detachFromContext() noexcept {
        WHERE_AM_I()
    }

    //下面是Creator的类函数定义
    PluginFieldCollection AffineTransPluginCreator::fc_{};
    std::vector<PluginField> AffineTransPluginCreator::attr_;

    AffineTransPluginCreator::AffineTransPluginCreator() {
        WHERE_AM_I()
        fc_.nbFields = attr_.size();
        fc_.fields   = attr_.data();
    }

    AffineTransPluginCreator::~AffineTransPluginCreator() noexcept {
        WHERE_AM_I()
    }

    //Creator里最重要的两个函数，分别用于“接受参数创建 Plugin” 和 “去序列化创建 Plugin”
    IPluginV2 *AffineTransPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept {
        WHERE_AM_I()
        float k = 2;
        float b = 1;  //这个作为默认的参数？
        std::map<std::string,float *> parameterMap{
                {"k",&k},
                {"b",&b}
        };
        for (int i = 0; i < fc->nbFields; ++i)
        {
            if (parameterMap.find(fc->fields[i].name) != parameterMap.end())
            {
                *parameterMap[fc->fields[i].name] = *reinterpret_cast<const float *>(fc->fields[i].data);
            }
        }

        return new AffineTransPlugin(name,k,b);
    }

    IPluginV2 *AffineTransPluginCreator::deserializePlugin(const char *name, const void *serialData,
                                                          size_t serialLength) noexcept {
        WHERE_AM_I()
        return new AffineTransPlugin(name,serialData,serialLength);
    }

    void AffineTransPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept {
        WHERE_AM_I()
        namespace_ = pluginNamespace;
    }

    const char *AffineTransPluginCreator::getPluginNamespace() const noexcept {
        WHERE_AM_I()
        return namespace_.c_str();
    }

    const char *AffineTransPluginCreator::getPluginName() const noexcept {
        WHERE_AM_I()
        return PLUGIN_NAME;
    }

    const char *AffineTransPluginCreator::getPluginVersion() const noexcept {
        WHERE_AM_I()
        return PLUGIN_VERSION;
    }

    const PluginFieldCollection *AffineTransPluginCreator::getFieldNames() noexcept {
        WHERE_AM_I()
        return &fc_;
    }

    REGISTER_TENSORRT_PLUGIN(AffineTransPluginCreator);
} //namespace nvinfer1

