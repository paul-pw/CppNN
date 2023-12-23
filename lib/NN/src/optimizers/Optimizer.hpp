#pragma once
#include "tensor.hpp"

class Optimizer
{
public:
    Optimizer() = default;
    Optimizer(Optimizer &other) = default;
    virtual Tensor<double> calculate_update(Tensor<double> weight_tensor,
                                            Tensor<double> gradient_tensor) = 0;
};
