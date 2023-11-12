cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_COMPILER "nvcc")
# This is your project statement. You should always list languages;
# Listing the version is nice here since it sets lots of useful variables
project(
  parallel_hw3
  VERSION 1.0
  LANGUAGES CXX)

find_package(CUDA REQUIRED)

# Link each target with other targets or add options, etc.
include_directories(include)

# Adding something we can run - Output name matches target name
cuda_add_executable(benchmark src/benchmark.cu)


# Add tests
# include(FetchContent)
# FetchContent_Declare(
#  googletest
#  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
# )
# FetchContent_MakeAvailable(googletest)

# enable_testing()

# add_executable(hw3_tests tests/test_kernel.cpp)
# target_link_libraries(hw3_tests gtest_main)
# include(GoogleTest)
# gtest_discover_tests(hw3_tests)
