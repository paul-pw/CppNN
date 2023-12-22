#include "FullyConnected.hpp"

#include <memory>
#include <random>

#include "../helpers.hpp"

FullyConnected::FullyConnected(std::size_t input_size, std::size_t output_size, std::mt19937 &generator)
{
    m_weights = random_tensor({input_size, output_size}, 0.0, 1.0, generator);
    m_bias = random_tensor({1, output_size}, 0.0, 1.0, generator);
}

void FullyConnected::set_optimizer(std::unique_ptr<Optimizer> opt)
{
    m_optimizer = std::move(opt);
}
Tensor<double> FullyConnected::forward(Tensor<double> input_tensor)
{
    m_input_tensor = input_tensor;

}
Tensor<double> FullyConnected::backward(Tensor<double> error_tensor)
{
}
