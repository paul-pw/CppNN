file(GLOB mnist_io_files io/*)
add_executable(mnist_io ${mnist_io_files})
target_link_libraries(mnist_io PRIVATE tensor)
target_compile_features(mnist_io PUBLIC cxx_std_20 )

file(GLOB mnist_nn_files mnist/*)
add_executable(mnist_nn ${mnist_nn_files})
target_link_libraries(mnist_nn PRIVATE tensor NN)
target_compile_features(mnist_nn PUBLIC cxx_std_20 )

