#pragma once
#include <memory>

#include "matvec.hpp"
#include "tensor.hpp"

class Optimizer
{
public:
    Optimizer() = default;
    virtual void update(Matrix<double>& weight_tensor, const Matrix<double>& gradient_tensor) = 0;
    virtual void update(Vector<double>& bias_tensor, const Vector<double>& gradient_tensor) = 0;
    virtual std::unique_ptr<Optimizer> clone() = 0;
    virtual ~Optimizer() = default;
};
