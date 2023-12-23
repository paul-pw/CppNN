#pragma once
#include <matvec.hpp>
#include <memory>

#include "../optimizers/Optimizer.hpp"

class BaseLayer
{
public:
    virtual Matrix<double> forward(const Matrix<double> &input_tensor) = 0;
    virtual Matrix<double> backward(const Matrix<double> &error_tensor) = 0;
    virtual void set_optimizer(std::unique_ptr<Optimizer> opt) = 0;
    virtual ~BaseLayer() = default;
};
