#include "ReLU.hpp"

#include <helpers.hpp>

Matrix<double> ReLU::forward(const Matrix<double> &input_tensor)
{
    m_input_tensor = input_tensor;
    return map(input_tensor, [](auto val) { return std::max(val, 0.0); });
}

Matrix<double> ReLU::backward(const Matrix<double> &error_tensor)
{
    return map(m_input_tensor, error_tensor,
               [](auto input, auto error) { return input <= 0 ? 0 : error; });
}
