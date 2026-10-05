#pragma once
// 测试向量读取工具：meta.txt (key=value) + raw bin
// bin 格式约定（与 tools/gen_test_vectors.py 严格一致）：
//   pred.bin       : float32 raw, channel-major [C, N]，C = 4 + 1 + num_classes
//   golden_decode.bin / golden_nms.bin : int32 count, 随后 count * 6 个 float32
//                    每个检测 6 个数: x1, y1, x2, y2, score, (float)label
//   img.bin        : uint8 raw, HWC, 尺寸见 meta (img_w, img_h)
//   golden_pre.bin : float32 raw, CHW [3, dst_w, dst_h]

#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "yolo_types.h"

namespace yoloop {
namespace testio {

inline std::map<std::string, std::string> load_meta(const std::string& dir) {
    std::ifstream f(dir + "/meta.txt");
    if (!f) throw std::runtime_error("cannot open " + dir + "/meta.txt");
    std::map<std::string, std::string> m;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        m[line.substr(0, pos)] = line.substr(pos + 1);
    }
    return m;
}

inline int meta_int(const std::map<std::string, std::string>& m, const std::string& k) {
    return std::stoi(m.at(k));
}
inline float meta_float(const std::map<std::string, std::string>& m, const std::string& k) {
    return std::stof(m.at(k));
}

template <typename T>
inline std::vector<T> load_raw(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.seekg(0, std::ios::end);
    size_t bytes = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<T> v(bytes / sizeof(T));
    f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)bytes);
    return v;
}

inline std::vector<float> load_f32(const std::string& path) { return load_raw<float>(path); }
inline std::vector<uint8_t> load_u8(const std::string& path) { return load_raw<uint8_t>(path); }

inline std::vector<Detection> load_dets(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    int32_t count = 0;
    f.read(reinterpret_cast<char*>(&count), 4);
    std::vector<Detection> out(count);
    for (int32_t i = 0; i < count; ++i) {
        float buf[6];
        f.read(reinterpret_cast<char*>(buf), 24);
        out[i] = {buf[0], buf[1], buf[2], buf[3], buf[4], (int32_t)buf[5]};
    }
    return out;
}

}  // namespace testio
}  // namespace yoloop
