#pragma once

#include "../optimizers/Optimizer.hpp"
#include "BaseLayer.hpp"
#include "tensor.hpp"

class FullyConnected : BaseLayer
{
public:
    FullyConnected(std::size_t input_size, std::size_t output_size);
    Tensor<double> forward(Tensor<double> input_tensor) override;
    Tensor<double> backward(Tensor<double> error_tensor) override;
    ~FullyConnected() override = default;
private:
    Tensor<double> m_weights;
    std::unique_ptr<Optimizer> m_optimizer;
    Tensor<double> m_input_tensor;
};
