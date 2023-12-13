cmake_minimum_required(VERSION 3.14)

project(project_mnist VERSION 1.0
        DESCRIPTION "TensorTech Pioneers Mnist Project"
        LANGUAGES CXX)

set(CXX_STANDARD 20)

######## GTEST

# for Gtest from https://google.github.io/googletest/quickstart-cmake.html
include(FetchContent)
FetchContent_Declare(
  googletest
  DOWNLOAD_EXTRACT_TIMESTAMP false
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()
include(GoogleTest)

####### GTEST

add_subdirectory(lib)
add_subdirectory(src)
