add_executable(yolo_rt
  main.cpp
  ncnn_infer.cpp
  tracker.cpp
  control_tx.cpp)
target_link_libraries(yolo_rt PRIVATE yoloop_x86 yoloop_common ncnn)
target_include_directories(yolo_rt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/x86)
target_compile_options(yolo_rt PRIVATE -O3 -ffast-math -march=${YOLOOP_X86_ARCH})
if(OpenMP_CXX_FOUND)
  target_link_libraries(yolo_rt PRIVATE OpenMP::OpenMP_CXX)
endif()
find_package(Threads REQUIRED)
target_link_libraries(yolo_rt PRIVATE Threads::Threads)

add_executable(test_tracker tests/test_tracker.cpp tracker.cpp)
target_link_libraries(test_tracker PRIVATE yoloop_common)
target_include_directories(test_tracker PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_queue tests/test_queue.cpp)
target_link_libraries(test_queue PRIVATE yoloop_common Threads::Threads)
target_include_directories(test_queue PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# 视频基准（可选，需 OpenCV）
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
  add_executable(bench_pipeline bench_pipeline.cpp ncnn_infer.cpp tracker.cpp)
  target_link_libraries(bench_pipeline PRIVATE yoloop_x86 yoloop_common ncnn
                        Threads::Threads ${OpenCV_LIBS})
  target_include_directories(bench_pipeline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                             ${CMAKE_SOURCE_DIR}/x86)
  target_compile_options(bench_pipeline PRIVATE -O3 -ffast-math -march=${YOLOOP_X86_ARCH})
else()
  message(STATUS "未找到 OpenCV, 跳过 bench_pipeline")
endif()
