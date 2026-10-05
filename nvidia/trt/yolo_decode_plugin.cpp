// YoloDecodePlugin 实现: 序列化格式(顺序固定)
//   int32 num_classes, float32 conf_thr, int32 input_w, int32 input_h,
//   int32 max_output_boxes, int32 n, 随后 5n 个 float32 (gx|gy|stride|aw|ah)
#include "yolo_decode_plugin.h"

#include <cuda_runtime.h>

#include <stdexcept>

#include "../kernel_launch.h"

namespace yoloop {
namespace trt {

namespace {
constexpr const char* kPluginName = "YoloDecode";
constexpr const char* kPluginVersion = "1";

template <typename T>
void writeTo(char*& p, const T& v) {
    std::memcpy(p, &v, sizeof(T));
    p += sizeof(T);
}

template <typename T>
T readFrom(const char*& p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
}
}  // namespace

YoloDecodePlugin::YoloDecodePlugin(int num_classes, float conf_thr,
                                   int input_w, int input_h, int max_output_boxes)
    : num_classes_(num_classes),
      conf_thr_(conf_thr),
      input_w_(input_w),
      input_h_(input_h),
      max_output_boxes_(max_output_boxes) {
    anchor_host_ = build_anchor_table(default_levels(input_w, input_h));
    upload_anchors();
}

YoloDecodePlugin::YoloDecodePlugin(const void* data, size_t length) {
    const char* p = static_cast<const char*>(data);
    const char* end = p + length;
    num_classes_ = readFrom<int32_t>(p);
    conf_thr_ = readFrom<float>(p);
    input_w_ = readFrom<int32_t>(p);
    input_h_ = readFrom<int32_t>(p);
    max_output_boxes_ = readFrom<int32_t>(p);
    const int32_t n = readFrom<int32_t>(p);
    if (n <= 0 || p + (size_t)5 * n * sizeof(float) > end) return;  // 数据损坏
    anchor_host_.n = n;
    const size_t bytes = (size_t)n * sizeof(float);
    anchor_host_.gx.resize(n);
    anchor_host_.gy.resize(n);
    anchor_host_.stride.resize(n);
    anchor_host_.aw.resize(n);
    anchor_host_.ah.resize(n);
    std::memcpy(anchor_host_.gx.data(), p, bytes);
    p += bytes;
    std::memcpy(anchor_host_.gy.data(), p, bytes);
    p += bytes;
    std::memcpy(anchor_host_.stride.data(), p, bytes);
    p += bytes;
    std::memcpy(anchor_host_.aw.data(), p, bytes);
    p += bytes;
    std::memcpy(anchor_host_.ah.data(), p, bytes);
    upload_anchors();
}

YoloDecodePlugin::~YoloDecodePlugin() {
    if (anchor_buf_) cudaFree(anchor_buf_);
    if (scratch_dets_) cudaFree(scratch_dets_);
    if (scratch_count_) cudaFree(scratch_count_);
}

void YoloDecodePlugin::upload_anchors() {
    const int n = anchor_host_.n;
    if (n <= 0 || max_output_boxes_ <= 0) return;
    const size_t bytes = (size_t)n * sizeof(float);
    std::vector<float> tmp((size_t)5 * n);
    std::memcpy(tmp.data(), anchor_host_.gx.data(), bytes);
    std::memcpy(tmp.data() + n, anchor_host_.gy.data(), bytes);
    std::memcpy(tmp.data() + 2 * n, anchor_host_.stride.data(), bytes);
    std::memcpy(tmp.data() + 3 * n, anchor_host_.aw.data(), bytes);
    std::memcpy(tmp.data() + 4 * n, anchor_host_.ah.data(), bytes);
    if (cudaMalloc(&anchor_buf_, 5 * bytes) != cudaSuccess) return;
    if (cudaMemcpy(anchor_buf_, tmp.data(), 5 * bytes, cudaMemcpyHostToDevice) != cudaSuccess)
        return;
    if (cudaMalloc(&scratch_dets_, (size_t)max_output_boxes_ * sizeof(Detection)) != cudaSuccess)
        return;
    if (cudaMalloc(&scratch_count_, sizeof(int)) != cudaSuccess) return;
    dev_tab_.gx = anchor_buf_;
    dev_tab_.gy = anchor_buf_ + n;
    dev_tab_.stride = anchor_buf_ + 2 * n;
    dev_tab_.aw = anchor_buf_ + 3 * n;
    dev_tab_.ah = anchor_buf_ + 4 * n;
    dev_tab_.n = n;
    ok_ = true;
}

const char* YoloDecodePlugin::getPluginType() const noexcept { return kPluginName; }
const char* YoloDecodePlugin::getPluginVersion() const noexcept { return kPluginVersion; }
int32_t YoloDecodePlugin::getNbOutputs() const noexcept { return 2; }
int32_t YoloDecodePlugin::initialize() noexcept { return ok_ ? 0 : 1; }
void YoloDecodePlugin::terminate() noexcept {}

size_t YoloDecodePlugin::getSerializationSize() const noexcept {
    return 5 * sizeof(int32_t) + sizeof(float) +
           (size_t)5 * anchor_host_.n * sizeof(float);
}

void YoloDecodePlugin::serialize(void* buffer) const noexcept {
    char* p = static_cast<char*>(buffer);
    writeTo<int32_t>(p, num_classes_);
    writeTo<float>(p, conf_thr_);
    writeTo<int32_t>(p, input_w_);
    writeTo<int32_t>(p, input_h_);
    writeTo<int32_t>(p, max_output_boxes_);
    writeTo<int32_t>(p, anchor_host_.n);
    const size_t bytes = (size_t)anchor_host_.n * sizeof(float);
    std::memcpy(p, anchor_host_.gx.data(), bytes);
    p += bytes;
    std::memcpy(p, anchor_host_.gy.data(), bytes);
    p += bytes;
    std::memcpy(p, anchor_host_.stride.data(), bytes);
    p += bytes;
    std::memcpy(p, anchor_host_.aw.data(), bytes);
    p += bytes;
    std::memcpy(p, anchor_host_.ah.data(), bytes);
}

void YoloDecodePlugin::destroy() noexcept { delete this; }
void YoloDecodePlugin::setPluginNamespace(const char* ns) noexcept { namespace_ = ns; }
const char* YoloDecodePlugin::getPluginNamespace() const noexcept { return namespace_.c_str(); }

nvinfer1::DataType YoloDecodePlugin::getOutputDataType(
    int32_t index, const nvinfer1::DataType* /*inputTypes*/, int32_t /*nbInputs*/) const noexcept {
    return index == 0 ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kINT32;
}

YoloDecodePlugin* YoloDecodePlugin::clone() const noexcept {
    try {
        std::vector<char> buf(getSerializationSize());
        serialize(buf.data());
        return new YoloDecodePlugin(buf.data(), buf.size());
    } catch (...) {
        return nullptr;
    }
}

nvinfer1::DimsExprs YoloDecodePlugin::getOutputDimensions(
    int32_t outputIndex, const nvinfer1::DimsExprs* /*inputs*/, int32_t /*nbInputs*/,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
    nvinfer1::DimsExprs out{};
    if (outputIndex == 0) {
        out.nbDims = 3;
        out.d[0] = exprBuilder.constant(1);
        out.d[1] = exprBuilder.constant(max_output_boxes_);
        out.d[2] = exprBuilder.constant(6);
    } else {
        out.nbDims = 2;
        out.d[0] = exprBuilder.constant(1);
        out.d[1] = exprBuilder.constant(1);
    }
    return out;
}

bool YoloDecodePlugin::supportsFormatCombination(
    int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t /*nbInputs*/,
    int32_t /*nbOutputs*/) noexcept {
    if (inOut[pos].format != nvinfer1::TensorFormat::kLINEAR) return false;
    // pos 0: 输入 [1,C,N] float; pos 1: dets [1,cap,6] float; pos 2: count [1,1] int32
    if (pos == 2) return inOut[pos].type == nvinfer1::DataType::kINT32;
    return inOut[pos].type == nvinfer1::DataType::kFLOAT;
}

void YoloDecodePlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* /*in*/, int32_t /*nbInputs*/,
    const nvinfer1::DynamicPluginTensorDesc* /*out*/, int32_t /*nbOutputs*/) noexcept {
    // 配置(类别数/anchor 表/cap)已在构造与序列化中固定, 无需按 shape 调整
}

size_t YoloDecodePlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* /*inputs*/,
                                          int32_t /*nbInputs*/,
                                          const nvinfer1::PluginTensorDesc* /*outputs*/,
                                          int32_t /*nbOutputs*/) const noexcept {
    return 0;  // scratch 为插件内部持有
}

int32_t YoloDecodePlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                  const nvinfer1::PluginTensorDesc* /*outputDesc*/,
                                  const void* const* inputs, void* const* outputs,
                                  void* /*workspace*/, cudaStream_t stream) noexcept {
    if (!ok_) return 1;
    try {
        // 校验输入维度: [1, 4+1+num_classes, n]
        const nvinfer1::Dims& d = inputDesc[0].dims;
        if (d.nbDims != 3 || d.d[1] != 4 + 1 + num_classes_ || d.d[2] != dev_tab_.n)
            return 1;
        const float* pred = static_cast<const float*>(inputs[0]);
        float* out_dets = static_cast<float*>(outputs[0]);
        int* out_count = static_cast<int*>(outputs[1]);
        cuda::decode(pred, dev_tab_, num_classes_, conf_thr_,
                     scratch_dets_, scratch_count_, max_output_boxes_, stream);
        cuda::detail::launch_dets_to_f6(scratch_dets_, scratch_count_,
                                        out_dets, out_count, max_output_boxes_, stream);
        return 0;
    } catch (...) {
        return 1;
    }
}

// ---------------- Creator ----------------

YoloDecodePluginCreator::YoloDecodePluginCreator() {
    fields_.emplace_back("num_classes", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    fields_.emplace_back("conf_thr", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1);
    fields_.emplace_back("input_w", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    fields_.emplace_back("input_h", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    fields_.emplace_back("max_output_boxes", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    fc_.nbFields = static_cast<int32_t>(fields_.size());
    fc_.fields = fields_.data();
}

const char* YoloDecodePluginCreator::getPluginName() const noexcept { return kPluginName; }
const char* YoloDecodePluginCreator::getPluginVersion() const noexcept { return kPluginVersion; }

const nvinfer1::PluginFieldCollection* YoloDecodePluginCreator::getFieldNames() noexcept {
    return &fc_;
}

nvinfer1::IPluginV2* YoloDecodePluginCreator::createPlugin(
    const char* /*name*/, const nvinfer1::PluginFieldCollection* fc) noexcept {
    try {
        int num_classes = 80, input_w = 640, input_h = 640, max_boxes = 1024;
        float conf_thr = 0.25f;
        for (int32_t i = 0; i < fc->nbFields; ++i) {
            const nvinfer1::PluginField& f = fc->fields[i];
            if (!std::strcmp(f.name, "num_classes"))
                num_classes = *static_cast<const int32_t*>(f.data);
            else if (!std::strcmp(f.name, "conf_thr"))
                conf_thr = *static_cast<const float*>(f.data);
            else if (!std::strcmp(f.name, "input_w"))
                input_w = *static_cast<const int32_t*>(f.data);
            else if (!std::strcmp(f.name, "input_h"))
                input_h = *static_cast<const int32_t*>(f.data);
            else if (!std::strcmp(f.name, "max_output_boxes"))
                max_boxes = *static_cast<const int32_t*>(f.data);
        }
        return new YoloDecodePlugin(num_classes, conf_thr, input_w, input_h, max_boxes);
    } catch (...) {
        return nullptr;
    }
}

nvinfer1::IPluginV2* YoloDecodePluginCreator::deserializePlugin(
    const char* /*name*/, const void* serialData, size_t serialLength) noexcept {
    try {
        return new YoloDecodePlugin(serialData, serialLength);
    } catch (...) {
        return nullptr;
    }
}

void YoloDecodePluginCreator::setPluginNamespace(const char* ns) noexcept { namespace_ = ns; }
const char* YoloDecodePluginCreator::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

}  // namespace trt
}  // namespace yoloop

// REGISTER_TENSORRT_PLUGIN 宏会做 token 粘贴, 模板实参必须是非限定名
using yoloop::trt::YoloDecodePluginCreator;
REGISTER_TENSORRT_PLUGIN(YoloDecodePluginCreator);
