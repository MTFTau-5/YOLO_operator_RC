// TRT 插件最小 sanity: 创建 -> serialize -> creator 反序列化 -> enqueue 实跑 -> 销毁
// 输入 pred 全 -10, 前 7 个 anchor 置高 obj/cls, 期望恰好 7 个检测
#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "yolo_decode_plugin.h"

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                     \
                    cudaGetErrorString(err_), __FILE__, __LINE__);           \
            return 1;                                                        \
        }                                                                    \
    } while (0)

int main() {
    using namespace nvinfer1;
    constexpr int kNumClasses = 80;
    constexpr int kCap = 100;
    constexpr float kConfThr = 0.25f;

    yoloop::trt::YoloDecodePluginCreator creator;
    int num_classes = kNumClasses, input_w = 640, input_h = 640, max_boxes = kCap;
    float conf_thr = kConfThr;
    PluginField fields[5] = {
        {"num_classes", &num_classes, PluginFieldType::kINT32, 1},
        {"conf_thr", &conf_thr, PluginFieldType::kFLOAT32, 1},
        {"input_w", &input_w, PluginFieldType::kINT32, 1},
        {"input_h", &input_h, PluginFieldType::kINT32, 1},
        {"max_output_boxes", &max_boxes, PluginFieldType::kINT32, 1},
    };
    PluginFieldCollection fc{5, fields};

    IPluginV2* raw = creator.createPlugin("YoloDecode", &fc);
    if (!raw) {
        fprintf(stderr, "createPlugin failed\n");
        return 1;
    }
    auto* plugin = dynamic_cast<yoloop::trt::YoloDecodePlugin*>(raw);
    if (!plugin || plugin->getNbOutputs() != 2 ||
        std::strcmp(plugin->getPluginType(), "YoloDecode") != 0 ||
        plugin->getOutputDataType(0, nullptr, 0) != DataType::kFLOAT ||
        plugin->getOutputDataType(1, nullptr, 0) != DataType::kINT32) {
        fprintf(stderr, "plugin metadata mismatch\n");
        return 1;
    }
    printf("created: type=%s version=%s nbOutputs=%d\n", plugin->getPluginType(),
           plugin->getPluginVersion(), plugin->getNbOutputs());

    // serialize -> deserialize
    std::vector<char> blob(plugin->getSerializationSize());
    plugin->serialize(blob.data());
    IPluginV2* raw2 = creator.deserializePlugin("YoloDecode", blob.data(), blob.size());
    if (!raw2) {
        fprintf(stderr, "deserializePlugin failed\n");
        return 1;
    }
    auto* plugin2 = dynamic_cast<yoloop::trt::YoloDecodePlugin*>(raw2);
    printf("serialize/deserialize OK (%zu bytes)\n", blob.size());

    // enqueue 实跑: pred [1, 85, 25200], 全 -10, 前 7 个 anchor 置高
    constexpr int C = 4 + 1 + kNumClasses;
    const int N = 25200;
    std::vector<float> pred((size_t)C * N, -10.f);
    for (int i = 0; i < 7; ++i) {
        pred[(size_t)4 * N + i] = 8.f;   // obj logit -> sigmoid ~ 0.9997
        pred[(size_t)5 * N + i] = 2.f;   // cls0 logit -> sigmoid ~ 0.8808
    }
    float* pred_dev = nullptr;
    float* out_dets = nullptr;
    int* out_count = nullptr;
    CUDA_CHECK(cudaMalloc(&pred_dev, pred.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out_dets, (size_t)kCap * 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out_count, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(pred_dev, pred.data(), pred.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    PluginTensorDesc in_desc{Dims{3, {1, C, N}}, DataType::kFLOAT, TensorFormat::kLINEAR};
    PluginTensorDesc out_desc[2] = {
        {Dims{3, {1, kCap, 6}}, DataType::kFLOAT, TensorFormat::kLINEAR},
        {Dims{2, {1, 1}}, DataType::kINT32, TensorFormat::kLINEAR}};
    const void* inputs[1] = {pred_dev};
    void* outputs[2] = {out_dets, out_count};

    if (plugin2->initialize() != 0) {
        fprintf(stderr, "initialize failed\n");
        return 1;
    }
    const int rc = plugin2->enqueue(&in_desc, out_desc, inputs, outputs, nullptr, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (rc != 0) {
        fprintf(stderr, "enqueue returned %d\n", rc);
        return 1;
    }
    int count = -1;
    CUDA_CHECK(cudaMemcpy(&count, out_count, sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<float> dets((size_t)kCap * 6);
    CUDA_CHECK(cudaMemcpy(dets.data(), out_dets, dets.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    printf("enqueue OK, count=%d\n", count);
    if (count != 7) {
        fprintf(stderr, "expected 7 detections, got %d\n", count);
        return 1;
    }
    const float expect_score = 1.f / (1.f + expf(-8.f)) * (1.f / (1.f + expf(-2.f)));
    for (int i = 0; i < count; ++i) {
        const float score = dets[(size_t)i * 6 + 4];
        const float label = dets[(size_t)i * 6 + 5];
        if (fabsf(score - expect_score) > 1e-3f || label != 0.f) {
            fprintf(stderr, "det %d: score=%.6f label=%.1f (expect ~%.6f, 0)\n",
                    i, score, label, expect_score);
            return 1;
        }
    }
    // cap 之后槽位应清零
    if (dets[(size_t)(kCap - 1) * 6 + 4] != 0.f) {
        fprintf(stderr, "tail slot not zeroed\n");
        return 1;
    }

    plugin2->terminate();
    plugin2->destroy();
    plugin->destroy();
    cudaFree(pred_dev);
    cudaFree(out_dets);
    cudaFree(out_count);
    printf("TRT plugin sanity PASS\n");
    return 0;
}
