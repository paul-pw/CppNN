#pragma once
#include <cstddef>
#include <functional>
#include <numeric>
#include <random>
#include <tensor.hpp>
#include <vector>
#include "matvec.hpp"

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

template <typename T>
Matrix<T> transpose(const Matrix<T>& tensor){
    Matrix<T> out{tensor.cols(), tensor.rows()};
    for(size_t i = 0; i<tensor.rows(); ++i){
        for(size_t j=0; j<tensor.cols(); ++j){
            out(j,i) = tensor(i,j);
        }
    }
    return out;
}

template <typename T>
Matrix<T> dot(const Matrix<T>& a, const Matrix<T>& b){
    Matrix<T> out{a.rows(), b.cols()};
    for(size_t i=0;i<a.rows(); ++i){
        for(size_t j=0; j<b.rows(); ++i){

        }
    }
}
