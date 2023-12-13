file(GLOB lib_src src/*)
add_library(tensor STATIC ${lib_src})
target_include_directories(tensor PUBLIC src/)
target_compile_features(tensor PUBLIC cxx_std_20 )

file(GLOB test_src tests/*)
add_executable(tensor_test ${test_src})
target_link_libraries(tensor_test GTest::gtest_main)
gtest_discover_tests(tensor_test)

