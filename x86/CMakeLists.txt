# 7840HS/x86 独立 C++ 算子库（不依赖第三方库；OpenMP 可选）
add_library(yoloop_x86 STATIC
  decode_avx2.cpp
  nms_simd.cpp
  preprocess_avx2.cpp
)
target_link_libraries(yoloop_x86 PUBLIC yoloop_common)
target_compile_options(yoloop_x86 PRIVATE -O3 -ffast-math -march=${YOLOOP_X86_ARCH})

if(OpenMP_CXX_FOUND)
  target_link_libraries(yoloop_x86 PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(test_x86 tests/test_x86.cpp)
target_link_libraries(test_x86 PRIVATE yoloop_x86)
