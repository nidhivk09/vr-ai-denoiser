# Phase 8: LibTorch integration. Configured from the top-level CMakeLists.txt
# when find_package(Torch) succeeds.
add_library(pt_inference STATIC inference_pipeline.cpp)
target_include_directories(pt_inference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pt_inference PUBLIC pt_core ${TORCH_LIBRARIES})
# Recent LibTorch releases (2.10+) require C++20; older ones accept it too.
target_compile_features(pt_inference PUBLIC cxx_std_20)

add_executable(run_pipeline run_pipeline.cpp)
target_link_libraries(run_pipeline PRIVATE pt_inference)

if(PT_HAVE_CUDA AND NOT TORCH_CUDA_LIBRARIES AND NOT TARGET torch_cuda)
  message(WARNING "LibTorch looks like a CPU-only build; the pipeline will fail to run the denoiser on CUDA")
endif()

if(MSVC)
  # LibTorch DLLs must sit next to the executable on Windows.
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET run_pipeline POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:run_pipeline>)
endif()
