#pragma once

#include <filesystem>
#include <tensor.hpp>

int readInt32BE(std::ifstream &file);

Tensor<double> readidx3(std::filesystem::path path, size_t image_index);
Tensor<double> readidx1(std::filesystem::path path, size_t label_index);
