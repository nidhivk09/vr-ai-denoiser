# Header-only geometry tests: no CUDA, no LibTorch, no scene files, so they
# build and run anywhere (including the Mac dev machine). Run with `ctest`.
add_executable(test_stereo_geometry test_stereo_geometry.cpp)
target_include_directories(test_stereo_geometry PRIVATE ${CMAKE_SOURCE_DIR}/renderer/src)
target_compile_features(test_stereo_geometry PRIVATE cxx_std_17)
if(MSVC)
  target_compile_definitions(test_stereo_geometry PRIVATE _USE_MATH_DEFINES NOMINMAX)
endif()
add_test(NAME stereo_geometry COMMAND test_stereo_geometry)
