#pragma once

#include <cstddef>
#include <filesystem>
#include <tensor.hpp>
#include <vector>

int readInt32BE(std::ifstream &file);

Tensor<double> readidx3(std::filesystem::path path, size_t image_index);
Tensor<double> readidx1(std::filesystem::path path, size_t label_index);

std::vector<Tensor<double>> readidx3_batches(std::filesystem::path path, size_t batch_size);
std::vector<Tensor<double>> readidx1_batches(std::filesystem::path path, size_t batch_size);
