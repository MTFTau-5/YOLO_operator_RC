cmake_minimum_required(VERSION 3.24)
project(yolo_operator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(YOLOOP_BUILD_X86         "7840HS/x86 独立 C++ 算子" ON)
option(YOLOOP_BUILD_CUDA        "NVIDIA CUDA 算子"         ON)
option(YOLOOP_BUILD_TRT_PLUGIN  "TensorRT decode 插件"     ON)
option(YOLOOP_BUILD_PYTHON      "pybind11 Python 绑定"     ON)
option(YOLOOP_BUILD_VULKAN      "Vulkan 混合式 NMS(可选)"  ON)
# 目标机为 Zen4(7840HS) 时: -DYOLOOP_X86_ARCH=znver4
set(YOLOOP_X86_ARCH "native" CACHE STRING "x86 -march 目标")

find_package(OpenMP)

add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>)

# 公共标量参考实现 + 类型定义
add_library(yoloop_common STATIC common/yolo_ref.cpp)
target_include_directories(yoloop_common PUBLIC common)

if(YOLOOP_BUILD_X86)
  add_subdirectory(x86)
endif()

if(YOLOOP_BUILD_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    add_subdirectory(nvidia)
  else()
    message(STATUS "未找到 CUDA 编译器, 跳过 nvidia 模块")
  endif()
endif()

if(YOLOOP_BUILD_VULKAN)
  find_program(GLSLANG_VALIDATOR glslangValidator)
  find_package(Vulkan QUIET)
  if(GLSLANG_VALIDATOR AND Vulkan_FOUND)
    add_subdirectory(vulkan)
  else()
    message(STATUS "未找到 glslangValidator/Vulkan SDK, 跳过 vulkan 混合 NMS "
                   "(在目标机安装 glslang-tools 后自动启用)")
  endif()
endif()

message(STATUS "==================== yolo_operator 配置 ====================")
message(STATUS "  x86 模块:    ${YOLOOP_BUILD_X86} (march=${YOLOOP_X86_ARCH}, OpenMP=${OpenMP_CXX_FOUND})")
message(STATUS "  CUDA 模块:   ${YOLOOP_BUILD_CUDA} (compiler=${CMAKE_CUDA_COMPILER})")
message(STATUS "===========================================================")
