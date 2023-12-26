#include "SoftMax.hpp"

#include <cmath>
#include <helpers.hpp>

// TODO if we get unexpected results, implement input_shift before exp like in python.
Matrix<double> SoftMax::forward(const Matrix<double> &input_tensor)
{
    auto exponentiated = map(input_tensor, [](auto val) { return std::exp(val); });
    auto sum_exp = sum_axis(exponentiated, Axis::row);

    m_soft_max_result = Matrix<double>{exponentiated.rows(), exponentiated.cols()};

    //std::cout<< sum_exp.tensor();
    for (size_t i = 0; i < exponentiated.rows(); ++i)
    {
        for (size_t j = 0; j < exponentiated.cols(); ++j)
        {
            m_soft_max_result(i, j) = exponentiated(i, j) / sum_exp(i);
        }
    }
    return m_soft_max_result;
}

Matrix<double> SoftMax::backward(const Matrix<double> &error_tensor)
{
    auto error_prediction_sum = sum_axis(
        map(error_tensor, m_soft_max_result, [](auto a, auto b) { return a * b; }), Axis::row);
    
    Matrix<double> out{error_tensor.rows(), error_tensor.cols()};
    for (size_t i = 0; i < out.rows(); ++i)
    {
        for (size_t j = 0; j < out.cols(); ++j)
        {
            out(i,j) = m_soft_max_result(i,j)*(error_tensor(i,j) - error_prediction_sum(i));
        }
    }
    return out;
}
