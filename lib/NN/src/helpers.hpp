#pragma once
#include <cstddef>
#include <functional>
#include <numeric>
#include <random>
#include <tensor.hpp>
#include <vector>

// return a random tensor (for weight initialization for example)
template <typename T, typename Generator>
Tensor<T> random_tensor(const std::vector< size_t >& shape, T low, T high, Generator& prg_gen){
    Tensor<T> tensor{shape};
    auto size = std::accumulate(shape.begin(),shape.end(), 1, std::multiplies<T>());
    std::uniform_real_distribution<> dis{low, high};
    for(size_t i = 0; i<size; ++i){
        tensor(i) = dis(prg_gen);
    }
    return tensor;
}
