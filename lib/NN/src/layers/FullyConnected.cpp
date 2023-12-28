#include "FullyConnected.hpp"

#include <cmath>
#include <cstdlib>
#include <helpers.hpp>
#include <matvec.hpp>
#include <memory>
#include <random>

Tensor<double> xavier_initializer(const std::vector<size_t> &shape, double fan_in, double fan_out,
                                  std::mt19937 &gen)
{
    Tensor<double> tensor{shape};
    auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::normal_distribution dis{0.0, std::sqrt(2.0 / (fan_in + fan_out))};
    for (size_t i = 0; i < size; ++i)
    {
        tensor(i) = dis(gen);
    }
    return tensor;
}

Tensor<double> he_initializer(const std::vector<size_t> &shape, double fan_in, double fan_out,
                                  std::mt19937 &gen)
{
    Tensor<double> tensor{shape};
    auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::normal_distribution dis{0.0, std::sqrt(2.0 / (fan_in))};
    for (size_t i = 0; i < size; ++i)
    {
        tensor(i) = dis(gen);
    }
    return tensor;
}


// TODO maybe find a better initializing technique, -1,1 seems to give the best results for now.
FullyConnected::FullyConnected(std::size_t input_size, std::size_t output_size,
                               std::mt19937 &generator)
    : m_weights{he_initializer({input_size, output_size}, input_size, output_size, generator)},
    m_bias{he_initializer({output_size},input_size , output_size, generator)} 
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
