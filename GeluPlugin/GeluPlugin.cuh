#ifndef GELU_LIBRARY_CUH
#define GELU_LIBRARY_CUH

# include <NvInfer.h>
# include <cuda_fp16.h>
# include <string>
# include <vector>
# include <numeric>


#ifdef DEBUG
# define WHERE_AM_I()                           \
        do                                          \
        {                                           \
            printf("[%s]:this=%p\n",__func__,this); \
        }while(0);
#else
#define WHERE_AM_I()
#endif // ifdef DEBUG

# define CEIL_DIVIDE(X,Y) (((X)+(Y)-1)/(Y))
# define CEIL_TO(X,Y) (CEIL_DIVIDE(X,Y)*(Y))

namespace {
    static const char * PLUGIN_NAME {"Gelu"};
    static const char * PLUGIN_VERSION {"1"};
}

namespace nvinfer1{
    inline int64_t volume(const nvinfer1::Dims& d)
    {
        return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
    }

    class GeluPlugin:public IPluginV2DynamicExt{
    private:
        const std::string mLayerName;
        std::string mNamespace;

    public:
        GeluPlugin(const std::string name);
        GeluPlugin(const std::string name, const void* data, size_t length);
        GeluPlugin() = delete;
        // 从IPluginV2继承的函数
        const char  *getPluginType() const noexcept override;
        const char  *getPluginVersion() const noexcept override;
        int32_t     getNbOutputs() const noexcept override;        //返回多少Tensor
        int32_t     initialize() noexcept override;
        void        terminate() noexcept override;
        size_t      getSerializationSize() const noexcept override;
        void        serialize(void *buffer) const noexcept override;
        void        destroy() noexcept override;
        void        setPluginNamespace(const char *pluginNamespace) noexcept override;
        const char *getPluginNamespace() const noexcept override;
        //从IPluginV2Ext继承的函数
        DataType getOutputDataType(int32_t index, nvinfer1::DataType const *inputTypes, int32_t nbInputs) const noexcept override;
        void     attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept override;
        void     detachFromContext() noexcept override;
        //从IPluginV2DynamicExt继承的函数
        IPluginV2DynamicExt *clone() const noexcept override;
        DimsExprs            getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept override;
        bool                 supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override; //确定是否
        void                 configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept override;
        size_t               getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept override;
        int32_t              enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
        //enqueue才是实际执行op的函数，在其中调用kernel
    };

    class GeluPluginCreator:public IPluginCreator{
    private:
        static PluginFieldCollection fc_;
        static std::vector<PluginField> attr_;
        std::string namespace_;
    public:
        GeluPluginCreator();
        ~GeluPluginCreator();
        const char *                 getPluginName() const noexcept override;
        const char *                 getPluginVersion() const noexcept override;
        const PluginFieldCollection *getFieldNames() noexcept override;
        IPluginV2 *                  createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override;
        IPluginV2 *                  deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override;
        void                         setPluginNamespace(const char *pluginNamespace) noexcept override;
        const char *                 getPluginNamespace() const noexcept override;
    };
}

#endif //GELU_LIBRARY_CUH
