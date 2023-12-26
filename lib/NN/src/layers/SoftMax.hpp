#pragma once

#include "layers/BaseLayer.hpp"
#include "matvec.hpp"

class SoftMax : public BaseLayer
{
public:
    SoftMax() = default;
    Matrix<double> forward(const Matrix<double> &input_tensor) override;
    Matrix<double> backward(const Matrix<double> &error_tensor) override;

private:
    Matrix<double> m_soft_max_result;
};
