file(GLOB_RECURSE lib_src src/*)
add_library(NN STATIC ${lib_src})
target_include_directories(NN PUBLIC src/)
target_link_libraries(NN PUBLIC tensor) 
target_compile_features(NN PUBLIC cxx_std_20 )
#target_link_libraries(NN PUBLIC OpenMP::OpenMP_CXX)

file(GLOB_RECURSE test_src tests/*)
add_executable(NN_test ${test_src})
target_link_libraries(NN_test GTest::gtest_main)
target_link_libraries(NN_test NN)
gtest_discover_tests(NN_test)

