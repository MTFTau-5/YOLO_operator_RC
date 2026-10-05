# NVIDIA CUDA 算子模块: yoloop_cuda 静态库 + test_cuda + pybind11 绑定 + TRT 插件(可选)

find_package(CUDAToolkit REQUIRED)

# ---- 核心静态库 ----
add_library(yoloop_cuda STATIC
  decode_kernel.cu
  nms_bitmask_kernel.cu
  preprocess_kernel.cu
  yolo_ops.cu)
target_link_libraries(yoloop_cuda PUBLIC yoloop_common CUDA::cudart)
target_include_directories(yoloop_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(yoloop_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>)
# 静态库会被链进 shared 目标(pybind 模块 / TRT 插件), 需要 PIC
set_target_properties(yoloop_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(yoloop_common PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- 验证/基准程序 ----
add_executable(test_cuda tests/test_cuda.cu)
target_link_libraries(test_cuda PRIVATE yoloop_cuda yoloop_common)
target_compile_options(test_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>)

# ---- Python 绑定 ----
if(YOLOOP_BUILD_PYTHON)
  find_program(YOLOOP_PYTHON NAMES python python3)
  if(NOT YOLOOP_PYTHON)
    message(STATUS "未找到 python, 跳过 pybind11 绑定")
  else()
    execute_process(
      COMMAND ${YOLOOP_PYTHON} -c "import pybind11; print(pybind11.get_cmake_dir())"
      OUTPUT_VARIABLE YOLOOP_PYBIND11_CMAKE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if(YOLOOP_PYBIND11_CMAKE_DIR)
      find_package(pybind11 CONFIG PATHS "${YOLOOP_PYBIND11_CMAKE_DIR}" NO_DEFAULT_PATH)
    endif()
    if(pybind11_FOUND)
      pybind11_add_module(yolo_ops_cuda bindings/pybind_module.cpp)
      target_link_libraries(yolo_ops_cuda PRIVATE yoloop_cuda)
      message(STATUS "pybind11 绑定启用: ${YOLOOP_PYBIND11_CMAKE_DIR}")
    else()
      message(STATUS "未找到 pybind11 cmake 配置, 跳过 Python 绑定")
    endif()
  endif()
endif()

# ---- TensorRT decode 插件 (pip 包无 C++ 头文件, 需自行下载到 nvidia/trt/include) ----
if(YOLOOP_BUILD_TRT_PLUGIN)
  if(NOT YOLOOP_PYTHON)
    find_program(YOLOOP_PYTHON NAMES python python3)
  endif()
  # tensorrt pip 包的 site-packages 路径(头文件 hint 与动态库目录)
  set(YOLOOP_SP "")
  if(YOLOOP_PYTHON)
    execute_process(
      COMMAND ${YOLOOP_PYTHON} -c "import tensorrt, os; print(os.path.dirname(os.path.dirname(tensorrt.__file__)))"
      OUTPUT_VARIABLE YOLOOP_SP
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
  endif()
  find_path(TRT_INCLUDE_DIR NvInfer.h
    HINTS ${CMAKE_CURRENT_SOURCE_DIR}/trt/include
          ${YOLOOP_SP}/tensorrt/include
    PATH_SUFFIXES include)
  find_library(TRT_NVINFER_LIB
    NAMES libnvinfer.so.11 nvinfer
    HINTS ${YOLOOP_SP}/tensorrt_libs ${YOLOOP_SP}/tensorrt)
  if(TRT_INCLUDE_DIR AND TRT_NVINFER_LIB)
    add_library(yoloop_trt_plugins SHARED trt/yolo_decode_plugin.cpp)
    target_include_directories(yoloop_trt_plugins PRIVATE ${TRT_INCLUDE_DIR})
    target_link_libraries(yoloop_trt_plugins PRIVATE yoloop_cuda CUDA::cudart ${TRT_NVINFER_LIB})
    get_filename_component(TRT_LIBS_DIR "${TRT_NVINFER_LIB}" DIRECTORY)
    set_target_properties(yoloop_trt_plugins PROPERTIES
      BUILD_RPATH "${TRT_LIBS_DIR}" INSTALL_RPATH "${TRT_LIBS_DIR}")
    # IPluginV2DynamicExt 在 TRT 11 已标记 deprecated, 局部抑制弃用告警
    target_compile_options(yoloop_trt_plugins PRIVATE -Wno-deprecated-declarations)

    add_executable(test_trt_plugin trt/test_trt_plugin.cpp)
    target_include_directories(test_trt_plugin PRIVATE ${TRT_INCLUDE_DIR})
    target_link_libraries(test_trt_plugin PRIVATE yoloop_trt_plugins CUDA::cudart ${TRT_NVINFER_LIB})
    target_compile_options(test_trt_plugin PRIVATE -Wno-deprecated-declarations)
    set_target_properties(test_trt_plugin PROPERTIES
      BUILD_RPATH "${TRT_LIBS_DIR}" INSTALL_RPATH "${TRT_LIBS_DIR}")
    message(STATUS "TensorRT 插件启用: NvInfer.h=${TRT_INCLUDE_DIR}, nvinfer=${TRT_NVINFER_LIB}")
  else()
    message(STATUS "未找到 TensorRT 头文件(NvInfer.h)或 libnvinfer, 跳过 TRT 插件 "
                   "(可从 https://github.com/NVIDIA/TensorRT 对应 tag 下载 include/ "
                   "到 nvidia/trt/include 后自动启用)")
  endif()
endif()
