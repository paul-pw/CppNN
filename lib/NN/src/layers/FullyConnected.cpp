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

void FullyConnected::set_optimizer(const std::unique_ptr<Optimizer> &opt)
{
    m_weight_optimizer = opt->clone();
    m_bias_optimizer = opt->clone();
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
    // calculate gradient update
    auto input_tensor_T = transpose(m_input_tensor);
    auto gradient_weights = dot(input_tensor_T, error_tensor);
    auto gradient_bias = sum_axis(error_tensor, Axis::col); // TODO possibly wrong, use Axis::row
    if (m_weight_optimizer != nullptr && m_bias_optimizer != nullptr)
    {
        m_weight_optimizer->update(m_weights, gradient_weights);
        m_bias_optimizer->update(m_bias, gradient_bias);
    }

    // calculate error tensor
    auto weights_T = transpose(m_weights);
    return dot(error_tensor, weights_T);
}
