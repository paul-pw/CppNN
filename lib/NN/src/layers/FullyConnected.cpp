#include "FullyConnected.hpp"

#include <helpers.hpp>
#include <matvec.hpp>
#include <memory>
#include <random>

FullyConnected::FullyConnected(std::size_t input_size, std::size_t output_size,
                               std::mt19937 &generator)
    : m_weights{random_tensor({input_size, output_size}, 0.0, 1.0, generator)},
      m_bias{random_tensor({output_size}, 0.0, 1.0, generator)}
{
}

void FullyConnected::set_optimizer(std::unique_ptr<Optimizer> opt)
{
    m_optimizer = std::move(opt);
}

Matrix<double> FullyConnected::forward(const Matrix<double> &input_tensor)
{
    m_input_tensor = input_tensor;
    auto out = dot(input_tensor, m_weights);
    add(out, m_bias, Axis::col);
    return out;
}

Matrix<double> FullyConnected::backward(const Matrix<double> &error_tensor)
{
    auto input_tensor_T = transpose(m_input_tensor);
    auto gradient_weights = dot(input_tensor_T, error_tensor);
    //auto gradient_bias = dot(ones, error_tensor);
}

FullyConnected::~FullyConnected(){};
