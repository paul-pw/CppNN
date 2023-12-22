#pragma once
#include <memory>
#include <tensor.hpp>
#include "../optimizers/Optimizer.hpp"

class BaseLayer
{
public:
    virtual Tensor<double> forward(Tensor<double> input_tensor) = 0;
    virtual Tensor<double> backward(Tensor<double> error_tensor) = 0;
    virtual void set_optimizer(std::unique_ptr<Optimizer> opt) = 0;
    virtual ~BaseLayer() = default;
};
