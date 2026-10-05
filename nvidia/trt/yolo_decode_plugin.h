#pragma once
// TensorRT decode 插件 (IPluginV2DynamicExt, TRT 11.x 中已 deprecated 但仍可用)
//   输入:  [1, C, N] float32, channel-major, C = 4 + 1 + num_classes
//   输出0: [1, max_output_boxes, 6] float32 (x1,y1,x2,y2,score,label), 尾部清零
//   输出1: [1, 1] int32, 实际检测数 = min(count, max_output_boxes)
// 配置 num_classes/conf_thr/input_w/input_h/max_output_boxes 与 anchor 表
// 一起序列化进 plugin; enqueue 调用 yoloop::cuda::decode kernel
#include <NvInfer.h>
#include <NvInferRuntimePlugin.h>

#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "../../common/yolo_types.h"
#include "../yolo_ops.h"

namespace yoloop {
namespace trt {

class YoloDecodePlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    YoloDecodePlugin(int num_classes, float conf_thr, int input_w, int input_h,
                     int max_output_boxes);
    // 反序列化构造: buffer 布局见 yolo_decode_plugin.cpp serialize()
    YoloDecodePlugin(const void* data, size_t length);
    ~YoloDecodePlugin() override;

    YoloDecodePlugin(const YoloDecodePlugin&) = delete;
    YoloDecodePlugin& operator=(const YoloDecodePlugin&) = delete;

    // IPluginV2
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* ns) noexcept override;
    const char* getPluginNamespace() const noexcept override;

    // IPluginV2Ext
    nvinfer1::DataType getOutputDataType(int32_t index,
                                         const nvinfer1::DataType* inputTypes,
                                         int32_t nbInputs) const noexcept override;

    // IPluginV2DynamicExt
    YoloDecodePlugin* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, const nvinfer1::PluginTensorDesc* inOut,
                                   int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs,
                         const nvinfer1::DynamicPluginTensorDesc* out,
                         int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs,
                            const nvinfer1::PluginTensorDesc* outputs,
                            int32_t nbOutputs) const noexcept override;
    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                    const nvinfer1::PluginTensorDesc* outputDesc,
                    const void* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override;

private:
    int num_classes_;
    float conf_thr_;
    int input_w_, input_h_;
    int max_output_boxes_;           // cap
    AnchorTable anchor_host_;        // host 侧 anchor 表(序列化来源)
    cuda::DeviceAnchorTable dev_tab_{};
    float* anchor_buf_ = nullptr;    // 5n floats
    Detection* scratch_dets_ = nullptr;   // cap 个 Detection
    int* scratch_count_ = nullptr;
    std::string namespace_;
    bool ok_ = false;

    void upload_anchors();           // 分配设备 buffer 并上传 anchor_host_
};

class YoloDecodePluginCreator : public nvinfer1::IPluginCreator {
public:
    YoloDecodePluginCreator();
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(
        const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* ns) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    std::string namespace_;
    nvinfer1::PluginFieldCollection fc_;
    std::vector<nvinfer1::PluginField> fields_;
};

}  // namespace trt
}  // namespace yoloop
