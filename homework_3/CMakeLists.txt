cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_COMPILER "/usr/local/cuda/bin/nvcc")
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
cuda_add_executable(benchmark 
  src/wide_matrix_multiplier.cupp 
  src/shuffle_single_warp_multiplier.cupp
  src/atomic_single_warp_multipler.cupp
  src/multiple_warp_multiplier.cupp
  src/benchmark.cpp )