
add_subdirectory(loadgen)

set(TARGET_NAME tvm_loadgen)
add_executable(${TARGET_NAME} benchmark.h classification.cpp classification.h main.cpp settings.h)
target_link_libraries(${TARGET_NAME} PRIVATE mlperf_loadgen tvm)
