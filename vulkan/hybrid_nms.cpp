#include "../common/yolo_types.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

#ifndef YOLOOP_VULKAN_SHADER
#define YOLOOP_VULKAN_SHADER "iou_matrix.spv"
#endif

namespace yoloop {
namespace vk {

namespace {

constexpr uint32_t kLocalSize = 16;  // 与 iou_matrix.comp 的 local_size_x/y 一致

struct Context {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipe_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

bool init_instance(Context& c) {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "yoloop_hybrid_nms";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, nullptr, &c.instance) != VK_SUCCESS) return false;

    uint32_t nphys = 0;
    vkEnumeratePhysicalDevices(c.instance, &nphys, nullptr);
    if (nphys == 0) return false;
    std::vector<VkPhysicalDevice> devs(nphys);
    vkEnumeratePhysicalDevices(c.instance, &nphys, devs.data());
    for (VkPhysicalDevice pd : devs) {
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> props(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, props.data());
        for (uint32_t q = 0; q < nq; ++q) {
            if (props[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                c.phys = pd;
                c.queue_family = q;
                return true;
            }
        }
    }
    return false;
}

bool init_device(Context& c) {
    float prio = 1.f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = c.queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    if (vkCreateDevice(c.phys, &dci, nullptr, &c.device) != VK_SUCCESS) return false;
    vkGetDeviceQueue(c.device, c.queue_family, 0, &c.queue);

    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = c.queue_family;
    if (vkCreateCommandPool(c.device, &cpci, nullptr, &c.cmd_pool) != VK_SUCCESS) return false;

    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &pool_size;
    return vkCreateDescriptorPool(c.device, &dpci, nullptr, &c.desc_pool) == VK_SUCCESS;
}

bool init_pipeline(Context& c, const char* spv_path) {
    VkDescriptorSetLayoutBinding bindings[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci{};
    dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = 3;
    dlci.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(c.device, &dlci, nullptr, &c.desc_layout) != VK_SUCCESS)
        return false;

    VkPushConstantRange pc_range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &c.desc_layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pc_range;
    if (vkCreatePipelineLayout(c.device, &plci, nullptr, &c.pipe_layout) != VK_SUCCESS)
        return false;

    FILE* f = std::fopen(spv_path, "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> spirv(sz > 0 ? (size_t)sz / sizeof(uint32_t) : 0);
    bool rd = sz > 0 && (sz % 4) == 0 &&
              std::fread(spirv.data(), 1, (size_t)sz, f) == (size_t)sz;
    std::fclose(f);
    if (!rd) return false;

    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = (size_t)sz;
    smci.pCode = spirv.data();
    VkShaderModule shader = VK_NULL_HANDLE;
    if (vkCreateShaderModule(c.device, &smci, nullptr, &shader) != VK_SUCCESS) return false;

    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = shader;
    cpci.stage.pName = "main";
    cpci.layout = c.pipe_layout;
    VkResult r =
        vkCreateComputePipelines(c.device, VK_NULL_HANDLE, 1, &cpci, nullptr, &c.pipeline);
    vkDestroyShaderModule(c.device, shader, nullptr);
    return r == VK_SUCCESS;
}

Context& ctx() {
    static Context c;
    static const bool inited =
        init_instance(c) && init_device(c) && init_pipeline(c, YOLOOP_VULKAN_SHADER);
    (void)inited;
    return c;
}

bool available() { return ctx().pipeline != VK_NULL_HANDLE; }

uint32_t find_host_memory(VkPhysicalDevice phys, uint32_t type_bits) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            return i;
    }
    return UINT32_MAX;
}

struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void* mapped = nullptr;
};

bool create_host_buffer(Context& c, VkDeviceSize size, Buffer& b) {
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(c.device, &bci, nullptr, &b.buf) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(c.device, b.buf, &req);
    uint32_t mt = find_host_memory(c.phys, req.memoryTypeBits);
    if (mt == UINT32_MAX) return false;
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = mt;
    if (vkAllocateMemory(c.device, &mai, nullptr, &b.mem) != VK_SUCCESS) return false;
    if (vkBindBufferMemory(c.device, b.buf, b.mem, 0) != VK_SUCCESS) return false;
    return vkMapMemory(c.device, b.mem, 0, size, 0, &b.mapped) == VK_SUCCESS;
}

void destroy_buffer(Context& c, Buffer& b) {
    if (b.mapped) vkUnmapMemory(c.device, b.mem);
    if (b.buf != VK_NULL_HANDLE) vkDestroyBuffer(c.device, b.buf, nullptr);
    if (b.mem != VK_NULL_HANDLE) vkFreeMemory(c.device, b.mem, nullptr);
    b = Buffer{};
}

// GPU 计算 N×N 上三角 IoU 矩阵（行主序写入 iou_out，下三角/对角保持 0）
bool gpu_iou_matrix(Context& c, const float* boxes4, const float* meta4, uint32_t n,
                    float* iou_out) {
    Buffer boxes, meta, iou;
    bool ok = create_host_buffer(c, (VkDeviceSize)n * 16, boxes) &&
              create_host_buffer(c, (VkDeviceSize)n * 16, meta) &&
              create_host_buffer(c, (VkDeviceSize)n * n * 4, iou);
    VkDescriptorSet dset = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    if (ok) {
        std::memcpy(boxes.mapped, boxes4, (size_t)n * 16);
        std::memcpy(meta.mapped, meta4, (size_t)n * 16);
        std::memset(iou.mapped, 0, (size_t)n * n * 4);  // shader 只写上三角

        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = c.desc_pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &c.desc_layout;
        ok = vkAllocateDescriptorSets(c.device, &dsai, &dset) == VK_SUCCESS;
    }
    if (ok) {
        VkDescriptorBufferInfo dbi[3] = {{boxes.buf, 0, VK_WHOLE_SIZE},
                                         {meta.buf, 0, VK_WHOLE_SIZE},
                                         {iou.buf, 0, VK_WHOLE_SIZE}};
        VkWriteDescriptorSet writes[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = dset;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &dbi[i];
        }
        vkUpdateDescriptorSets(c.device, 3, writes, 0, nullptr);

        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = c.cmd_pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        ok = vkAllocateCommandBuffers(c.device, &cbai, &cmd) == VK_SUCCESS;
    }
    if (ok) {
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        ok = vkCreateFence(c.device, &fci, nullptr, &fence) == VK_SUCCESS;
    }
    if (ok) {
        VkCommandBufferBeginInfo cbbi{};
        cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &cbbi);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, c.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, c.pipe_layout, 0, 1,
                                &dset, 0, nullptr);
        vkCmdPushConstants(cmd, c.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(uint32_t), &n);
        uint32_t groups = (n + kLocalSize - 1) / kLocalSize;
        vkCmdDispatch(cmd, groups, groups, 1);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        // HOST_COHERENT 内存：提交完成即对 host 可见，无需显式 invalidate
        ok = vkQueueSubmit(c.queue, 1, &si, fence) == VK_SUCCESS &&
             vkWaitForFences(c.device, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
    }
    if (ok) std::memcpy(iou_out, iou.mapped, (size_t)n * n * 4);

    if (fence != VK_NULL_HANDLE) vkDestroyFence(c.device, fence, nullptr);
    if (cmd != VK_NULL_HANDLE) vkFreeCommandBuffers(c.device, c.cmd_pool, 1, &cmd);
    if (dset != VK_NULL_HANDLE) vkFreeDescriptorSets(c.device, c.desc_pool, 1, &dset);
    destroy_buffer(c, boxes);
    destroy_buffer(c, meta);
    destroy_buffer(c, iou);
    return ok;
}

// CPU 位掩码贪心：iou 为已排序（score 降序）坐标下的 N×N 上三角矩阵
std::vector<Detection> bitmask_greedy(const std::vector<Detection>& dets,
                                      const std::vector<size_t>& order, const float* iou,
                                      size_t n, float nms_thr) {
    const size_t W = (n + 63) / 64;
    std::vector<uint64_t> row_mask(n * W, 0);  // row_mask[i]: i 抑制的 j 集合
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            if (iou[i * n + j] > nms_thr)
                row_mask[i * W + j / 64] |= 1ull << (j % 64);

    std::vector<uint64_t> sup(W, 0);
    std::vector<Detection> keep;
    keep.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if ((sup[i / 64] >> (i % 64)) & 1ull) continue;
        keep.push_back(dets[order[i]]);
        for (size_t w = 0; w < W; ++w) sup[w] |= row_mask[i * W + w];
    }
    return keep;
}

void cpu_iou_matrix(const std::vector<Detection>& dets, const std::vector<size_t>& order,
                    float* iou, size_t n) {
    std::memset(iou, 0, n * n * sizeof(float));
    for (size_t i = 0; i < n; ++i) {
        const Detection& a = dets[order[i]];
        float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
        for (size_t j = i + 1; j < n; ++j) {
            const Detection& b = dets[order[j]];
            if (a.label != b.label) continue;  // 已是 0
            float iw = std::max(std::min(a.x2, b.x2) - std::max(a.x1, b.x1), 0.f);
            float ih = std::max(std::min(a.y2, b.y2) - std::max(a.y1, b.y1), 0.f);
            float inter = iw * ih;
            float ua = area_a + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
            iou[i * n + j] = ua > 0.f ? inter / ua : 0.f;
        }
    }
}

}  // namespace

std::vector<Detection> hybrid_nms(std::vector<Detection> dets, float nms_thr, int topk) {
    const size_t n0 = dets.size();
    if (n0 == 0) return {};

    // score 降序（并列按原索引升序，与 ref::nms 的 stable 语义一致）
    std::vector<size_t> order(n0);
    std::iota(order.begin(), order.end(), (size_t)0);
    auto cmp = [&](size_t a, size_t b) {
        float sa = dets[a].score, sb = dets[b].score;
        if (sa != sb) return sa > sb;
        return a < b;
    };
    if (topk > 0 && n0 > (size_t)topk) {
        std::partial_sort(order.begin(), order.begin() + topk, order.end(), cmp);
        order.resize((size_t)topk);
    } else {
        std::sort(order.begin(), order.end(), cmp);
    }
    const size_t n = order.size();

    std::vector<float> iou(n * n, 0.f);
    bool done = false;
    if (available() && n > 0) {
        std::vector<float> boxes4(n * 4), meta4(n * 4);
        for (size_t k = 0; k < n; ++k) {
            const Detection& d = dets[order[k]];
            boxes4[k * 4 + 0] = d.x1; boxes4[k * 4 + 1] = d.y1;
            boxes4[k * 4 + 2] = d.x2; boxes4[k * 4 + 3] = d.y2;
            meta4[k * 4 + 0] = d.score;
            meta4[k * 4 + 1] = (float)d.label;
            meta4[k * 4 + 2] = (d.x2 - d.x1) * (d.y2 - d.y1);
            meta4[k * 4 + 3] = 0.f;
        }
        done = gpu_iou_matrix(ctx(), boxes4.data(), meta4.data(), (uint32_t)n, iou.data());
    }
    if (!done) cpu_iou_matrix(dets, order, iou.data(), n);  // 回退路径

    return bitmask_greedy(dets, order, iou.data(), n, nms_thr);
}

}  // namespace vk
}  // namespace yoloop
