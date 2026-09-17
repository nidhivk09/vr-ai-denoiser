cmake_minimum_required(VERSION 3.18)
project(vr_ai_denoiser LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/renderer/cmake/PtCuda.cmake)
add_subdirectory(renderer)

# Phase 8: LibTorch integration. Point CMake at LibTorch with
#   -DCMAKE_PREFIX_PATH=/path/to/libtorch
option(BUILD_INTEGRATION "Build the LibTorch render+denoise pipeline" ON)
if(BUILD_INTEGRATION)
  find_package(Torch QUIET)
  if(Torch_FOUND)
    add_subdirectory(integration)
  else()
    message(WARNING "LibTorch not found - skipping integration/ (set CMAKE_PREFIX_PATH to your libtorch folder)")
  endif()
endif()
