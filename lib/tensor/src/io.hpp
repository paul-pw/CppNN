#pragma once

#include <filesystem>
#include <tensor.hpp>

// TODO(Kirill) something like this
Tensor<double> readidx3(std::filesystem::path path);
Tensor<double> readidx1(std::filesystem::path path);
