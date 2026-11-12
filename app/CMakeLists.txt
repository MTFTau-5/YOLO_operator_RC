set(YOLOOP_SPACEMIT_ORT_LIBS)
set(YOLOOP_SPACEMIT_ORT_RPATH)
if(YOLOOP_BUILD_SPACEMIT_ORT)
  set(_ORT_INCLUDE_HINTS)
  set(_ORT_LIB_HINTS)
  if(SPACEMIT_ORT_ROOT)
    list(APPEND _ORT_INCLUDE_HINTS "${SPACEMIT_ORT_ROOT}/include")
    list(APPEND _ORT_LIB_HINTS "${SPACEMIT_ORT_ROOT}/lib" "${SPACEMIT_ORT_ROOT}/lib64")
  endif()
  find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS ${_ORT_INCLUDE_HINTS}
    PATHS /usr/include/onnxruntime /usr/include /usr/local/include /opt/spacemit/include)
  find_path(SPACEMIT_ORT_ENV_INCLUDE_DIR
    NAMES spacemit_ort_env.h
    HINTS ${_ORT_INCLUDE_HINTS} ${ONNXRUNTIME_INCLUDE_DIR}
    PATHS /usr/include /usr/local/include /opt/spacemit/include
    PATH_SUFFIXES onnxruntime)
  find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    HINTS ${_ORT_LIB_HINTS}
    PATHS /usr/lib /usr/local/lib /opt/spacemit/lib)
  find_library(SPACEMIT_EP_LIB
    NAMES spacemit_ep
    HINTS ${_ORT_LIB_HINTS}
    PATHS /usr/lib /usr/local/lib /opt/spacemit/lib)

  if(ONNXRUNTIME_INCLUDE_DIR AND SPACEMIT_ORT_ENV_INCLUDE_DIR AND
     ONNXRUNTIME_LIB AND SPACEMIT_EP_LIB)
    add_library(yoloop_spacemit_ort STATIC ort_infer.cpp)
    target_link_libraries(yoloop_spacemit_ort PUBLIC
      yoloop_common ${ONNXRUNTIME_LIB} ${SPACEMIT_EP_LIB})
    target_include_directories(yoloop_spacemit_ort PUBLIC
      ${ONNXRUNTIME_INCLUDE_DIR} ${SPACEMIT_ORT_ENV_INCLUDE_DIR})
    target_compile_definitions(yoloop_spacemit_ort PUBLIC YOLOOP_HAS_SPACEMIT_ORT=1)
    target_compile_options(yoloop_spacemit_ort PRIVATE -O2 ${YOLOOP_CPU_ARCH_OPTIONS})
    if(YOLOOP_RISCV_A100)
      target_compile_definitions(yoloop_spacemit_ort PRIVATE YOLOOP_RISCV_A100=1)
    endif()
    set(YOLOOP_SPACEMIT_ORT_LIBS yoloop_spacemit_ort)
    get_filename_component(_ORT_LIB_DIR "${ONNXRUNTIME_LIB}" DIRECTORY)
    get_filename_component(_EP_LIB_DIR "${SPACEMIT_EP_LIB}" DIRECTORY)
    list(APPEND YOLOOP_SPACEMIT_ORT_RPATH "${_ORT_LIB_DIR}" "${_EP_LIB_DIR}")
    list(REMOVE_DUPLICATES YOLOOP_SPACEMIT_ORT_RPATH)
    message(STATUS "SpaceMIT ONNX Runtime 后端启用: ${ONNXRUNTIME_LIB}; ${SPACEMIT_EP_LIB}")
  else()
    message(STATUS "未找到完整 SpaceMIT ONNX Runtime 开发文件，仅构建 ncnn 后端")
  endif()
endif()

add_executable(yolo_rt
  main.cpp
  infer_backend.cpp
  ncnn_infer.cpp
  tracker.cpp
  control_tx.cpp)
target_link_libraries(yolo_rt PRIVATE ${YOLOOP_CPU_LIBRARY} yoloop_common ncnn
                      ${YOLOOP_SPACEMIT_ORT_LIBS})
target_include_directories(yolo_rt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(yolo_rt PRIVATE -O3 -ffast-math ${YOLOOP_CPU_ARCH_OPTIONS})
if(OpenMP_CXX_FOUND)
  target_link_libraries(yolo_rt PRIVATE OpenMP::OpenMP_CXX)
endif()
find_package(Threads REQUIRED)
target_link_libraries(yolo_rt PRIVATE Threads::Threads)
set_property(TARGET yolo_rt APPEND PROPERTY BUILD_RPATH "${YOLOOP_SPACEMIT_ORT_RPATH}")

add_executable(test_tracker tests/test_tracker.cpp tracker.cpp)
target_link_libraries(test_tracker PRIVATE yoloop_common)
target_include_directories(test_tracker PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_queue tests/test_queue.cpp)
target_link_libraries(test_queue PRIVATE yoloop_common Threads::Threads)
target_include_directories(test_queue PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# 合成图集成测试：box-major 转置 / AfRaw 逐层探测（不依赖真实模型文件）
add_executable(test_head_layouts tests/test_head_layouts.cpp ncnn_infer.cpp)
target_link_libraries(test_head_layouts PRIVATE ${YOLOOP_CPU_LIBRARY} yoloop_common ncnn)
target_include_directories(test_head_layouts PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(test_head_layouts PRIVATE -O2)

# 真实模型多尺寸动态验收（case 由 tools/accept_raw_dynamic.py 生成，缺失时 SKIP）
add_executable(test_raw_model tests/test_raw_model.cpp ncnn_infer.cpp)
target_link_libraries(test_raw_model PRIVATE ${YOLOOP_CPU_LIBRARY} yoloop_common ncnn)
target_include_directories(test_raw_model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(test_raw_model PRIVATE -O2)

# 视频基准（可选，需 OpenCV；只链接用到的模块，避免 opencv_viz 拖入 VTK 依赖）
find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs videoio)
if(OpenCV_FOUND)
  add_executable(bench_pipeline bench_pipeline.cpp infer_backend.cpp ncnn_infer.cpp tracker.cpp)
  target_link_libraries(bench_pipeline PRIVATE ${YOLOOP_CPU_LIBRARY} yoloop_common ncnn
                        Threads::Threads ${OpenCV_LIBS} ${YOLOOP_SPACEMIT_ORT_LIBS})
  target_include_directories(bench_pipeline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_options(bench_pipeline PRIVATE -O3 -ffast-math ${YOLOOP_CPU_ARCH_OPTIONS})
  set_property(TARGET bench_pipeline APPEND PROPERTY BUILD_RPATH "${YOLOOP_SPACEMIT_ORT_RPATH}")
else()
  message(STATUS "未找到 OpenCV, 跳过 bench_pipeline")
endif()
