cmake_minimum_required(VERSION 3.18)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # Standalone build of the renderer only.
  project(vr_pathtracer LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
  endif()
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/PtCuda.cmake)
endif()

find_package(Threads REQUIRED)

add_library(pt_core STATIC
  src/scene_loader.cpp
  src/bvh.cpp
  src/image_io.cpp
  src/cpu_backend.cpp
)
target_include_directories(pt_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(pt_core PUBLIC cxx_std_17)
target_link_libraries(pt_core PUBLIC Threads::Threads)
if(MSVC)
  target_compile_definitions(pt_core PUBLIC _USE_MATH_DEFINES NOMINMAX)
endif()

if(PT_HAVE_CUDA)
  target_sources(pt_core PRIVATE src/pathtrace_kernel.cu)
  target_compile_definitions(pt_core PUBLIC PT_WITH_CUDA PT_USE_NVTX)
  target_link_libraries(pt_core PUBLIC CUDA::cudart)
  set_target_properties(pt_core PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
endif()

add_executable(pathtracer src/main.cpp)
target_link_libraries(pathtracer PRIVATE pt_core)
