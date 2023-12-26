#pragma once
#include <cassert>
#include <cstddef>
#include <functional>
#include <numeric>
#include <random>
#include <tensor.hpp>
#include <type_traits>
#include <vector>

#include "matvec.hpp"

// return a random tensor (for weight initialization for example)
template <typename T, typename Generator>
Tensor<T> random_tensor(const std::vector<size_t> &shape, T low, T high, Generator &prg_gen)
{
    Tensor<T> tensor{shape};
    auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<T>());
    std::uniform_real_distribution<> dis{low, high};
    for (size_t i = 0; i < size; ++i)
    {
        tensor(i) = dis(prg_gen);
    }
    return tensor;
}

// TODO performance optimizaton: implement transpose_and_dot
template <typename T> Matrix<T> transpose(const Matrix<T> &tensor)
{
    Matrix<T> out{tensor.cols(), tensor.rows()};
    for (size_t i = 0; i < tensor.rows(); ++i)
    {
        for (size_t j = 0; j < tensor.cols(); ++j)
        {
            out(j, i) = tensor(i, j);
        }
    }
    return out;
}

template <typename T> Matrix<T> dot(const Matrix<T> &a, const Matrix<T> &b)
{
    assert(a.cols() == b.rows());

    Matrix<T> out{a.rows(), b.cols()};

    for (size_t i = 0; i < a.rows(); ++i)
    {
        for (size_t j = 0; j < b.cols(); ++j)
        {
            for (size_t k = 0; k < b.rows(); ++k)
            {
                out(i, j) += a(i, k) * b(k, j);
            }
        }
    }
    return out;
}

enum class Axis
{
    row = 0,
    col,
};

// TODO potential optimization: switch rows and cols
// such that locality in underlying vector is best
// TODO put switch statement outside for for better performance?
// add colls
template <typename T> void add(Matrix<T> &m, const Vector<T> &v, Axis axis)
{
    if (axis == Axis::col)
    {
        assert(m.cols() == v.size());
    }
    else
    {
        assert(m.rows() == v.size());
    }

    for (size_t i = 0; i < m.rows(); ++i)
    {
        for (size_t j = 0; j < m.cols(); ++j)
        {
            switch (axis)
            {
            case Axis::col:
                m(i, j) += v(j);
                break;
            case Axis::row:
                m(i, j) += v(i);
                break;
            }
        }
    }
}

// sum along axis
template <typename T> Vector<T> sum_axis(const Matrix<T> &m, Axis axis)
{
    // vec_size depending on Axis
    auto vec_size = (axis == Axis::row) ? m.rows() : m.cols();
    Vector<T> out{vec_size};
    for (size_t i = 0; i < m.rows(); ++i)
    {
        for (size_t j = 0; j < m.cols(); ++j)
        {
            switch (axis)
            {
            case Axis::row:
                out(i) += m(i, j);
                break;
            case Axis::col:
                out(j) += m(i, j);
                break;
            }
        }
    }
    return out;
}


template <typename T,typename F> Matrix<T> map(const Matrix<T> &m, F func)
{
    Matrix<T> out{m.rows(), m.cols()};
    std::cout << out.rows()<< '\n';
    for (size_t i = 0; i < m.rows(); ++i)
    {
        for (size_t j = 0; j < m.cols(); ++j)
        {
            out(i, j) =  func(m(i, j));
        }
    }
    return out;
}

template <typename T, typename F> Matrix<T> map(const Matrix<T> &m1,const Matrix<T> &m2, F func)
{
    assert(m1.rows() == m2.rows());
    assert(m1.cols() == m2.cols());
    Matrix<T> out{m1.rows(), m1.cols()};
    for (size_t i = 0; i < m1.rows(); ++i)
    {
        for (size_t j = 0; j < m1.cols(); ++j)
        {
            out(i, j) = func(m1(i, j), m2(i,j));
        }
    }
    return out;
}
