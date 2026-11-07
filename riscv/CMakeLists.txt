add_library(yoloop_riscv STATIC
  decode_af_dist.cpp
  decode_af_raw.cpp
  decode_rvv.cpp
  nms_rvv.cpp
  preprocess_rvv.cpp
)
target_link_libraries(yoloop_riscv PUBLIC yoloop_common)
target_include_directories(yoloop_riscv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(yoloop_riscv PRIVATE -O3 -ffast-math)
target_compile_definitions(yoloop_riscv PUBLIC YOLOOP_CPU_BACKEND_RISCV=1)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
  target_compile_options(yoloop_riscv PUBLIC
    -march=${YOLOOP_RISCV_ARCH}
    -mabi=${YOLOOP_RISCV_ABI})
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(yoloop_riscv PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(test_rvv ../x86/tests/test_x86.cpp)
target_link_libraries(test_rvv PRIVATE yoloop_riscv)
