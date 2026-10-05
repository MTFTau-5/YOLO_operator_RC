# 混合式 NMS（README(1).md §8.5）：IoU 矩阵 compute shader + Vulkan host 端
# 仅在顶层检测到 glslangValidator 与 Vulkan SDK 时才会进入本目录；
# 本目录的任何文件都不影响默认（x86）构建。
set(YOLOOP_SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/iou_matrix.comp)
set(YOLOOP_SHADER_SPV ${CMAKE_CURRENT_BINARY_DIR}/iou_matrix.spv)

add_custom_command(
  OUTPUT ${YOLOOP_SHADER_SPV}
  COMMAND ${GLSLANG_VALIDATOR} -V -o ${YOLOOP_SHADER_SPV} ${YOLOOP_SHADER_SRC}
  DEPENDS ${YOLOOP_SHADER_SRC}
  COMMENT "glslangValidator: iou_matrix.comp -> iou_matrix.spv"
  VERBATIM)

add_library(yoloop_vulkan STATIC hybrid_nms.cpp ${YOLOOP_SHADER_SPV})
set_source_files_properties(${YOLOOP_SHADER_SPV} PROPERTIES GENERATED TRUE)
target_link_libraries(yoloop_vulkan PUBLIC yoloop_common Vulkan::Vulkan)
target_compile_definitions(yoloop_vulkan PRIVATE YOLOOP_VULKAN_SHADER="${YOLOOP_SHADER_SPV}")
