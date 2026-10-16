#pragma once
// 极简 YAML 子集解析（扁平 key: value，# 注释，[a,b] 或逗号分隔列表）。
// 工程零第三方依赖传统的延伸——配置只有一层，不引 yaml-cpp。
// 不支持的语法（嵌套/多行/引号转义）会在 value 里原样保留，由取值方校验报错。

#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace yoloop {

class Config {
public:
    // 读取扁平 YAML；语法错误（无冒号的非注释行）返回 false
    bool load(const std::string& path, std::string& err) {
        std::ifstream f(path);
        if (!f) {
            err = "打不开配置文件: " + path;
            return false;
        }
        std::string line;
        int lineno = 0;
        while (std::getline(f, line)) {
            lineno++;
            // 去注释（# 前无引号场景，扁平配置够用）
            size_t hash = line.find('#');
            if (hash != std::string::npos) line.erase(hash);
            auto trim = [](std::string& s) {
                size_t a = s.find_first_not_of(" \t\r");
                size_t b = s.find_last_not_of(" \t\r");
                s = (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
            };
            trim(line);
            if (line.empty()) continue;
            size_t colon = line.find(':');
            if (colon == std::string::npos) {
                err = path + ":" + std::to_string(lineno) + " 缺冒号: " + line;
                return false;
            }
            std::string k = line.substr(0, colon), v = line.substr(colon + 1);
            trim(k);
            trim(v);
            kv_[k] = v;
        }
        return true;
    }

    bool has(const std::string& k) const { return kv_.count(k) != 0; }

    std::string get_str(const std::string& k, const std::string& def) const {
        auto it = kv_.find(k);
        return it == kv_.end() ? def : it->second;
    }
    double get_num(const std::string& k, double def) const {
        auto it = kv_.find(k);
        if (it == kv_.end()) return def;
        return strtod(it->second.c_str(), nullptr);
    }
    bool get_bool(const std::string& k, bool def) const {
        auto it = kv_.find(k);
        if (it == kv_.end()) return def;
        return it->second == "true" || it->second == "1" || it->second == "yes";
    }
    // "[a, b, c]" 或 "a, b, c" → 字符串列表
    std::vector<std::string> get_list(const std::string& k) const {
        auto it = kv_.find(k);
        if (it == kv_.end()) return {};
        std::string v = it->second;
        if (!v.empty() && v.front() == '[' && v.back() == ']')
            v = v.substr(1, v.size() - 2);
        std::vector<std::string> out;
        std::stringstream ss(v);
        std::string item;
        while (std::getline(ss, item, ',')) {
            size_t a = item.find_first_not_of(" \t");
            size_t b = item.find_last_not_of(" \t");
            if (a != std::string::npos) out.push_back(item.substr(a, b - a + 1));
        }
        return out;
    }
    // 浮点列表（anchor 用）："10,13, 16,30, 33,23"
    std::vector<float> get_floats(const std::string& k) const {
        std::vector<float> out;
        for (auto& s : get_list(k)) out.push_back((float)strtod(s.c_str(), nullptr));
        return out;
    }

    const std::map<std::string, std::string>& all() const { return kv_; }

private:
    std::map<std::string, std::string> kv_;
};

}  // namespace yoloop
