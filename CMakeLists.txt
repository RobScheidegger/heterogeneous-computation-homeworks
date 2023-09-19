cmake_minimum_required(VERSION 3.14)

# This is your project statement. You should always list languages;
# Listing the version is nice here since it sets lots of useful variables
project(
  parallel_hw1
  VERSION 1.0
  LANGUAGES CXX)

# Link each target with other targets or add options, etc.
include_directories(include)

# Adding something we can run - Output name matches target name
add_executable(benchmark src/benchmark.cpp)