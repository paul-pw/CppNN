#include "SoftMax.hpp"

#include <algorithm>
#include <cmath>
#include <helpers.hpp>

// TODO if we get unexpected results, implement input_shift before exp like in python.
Matrix<double> SoftMax::forward(const Matrix<double> &input_tensor)
{
    Matrix<double> exponentiated{input_tensor.rows(), input_tensor.cols()};
    for(size_t i =0; i<input_tensor.rows(); ++i){
        double max = 0.0;
        for(size_t j=0; j<input_tensor.cols(); ++j){
            max = std::max(max, input_tensor(i,j));
        }
        //std::cout<<max<<'\n';
        for(size_t j = 0; j<input_tensor.cols(); ++j){
            //std::cout<<input_tensor(i,j)-max<< ';';
            exponentiated(i,j) = std::exp(input_tensor(i,j)-max);
        }
    }

    auto sum_exp = sum_axis(exponentiated, Axis::row);
    //std::cout<<exponentiated.tensor()<<'\n';

    m_soft_max_result = Matrix<double>{exponentiated.rows(), exponentiated.cols()};

    // std::cout<< sum_exp.tensor();
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

    //std::cout<<m_soft_max_result.tensor()<<'\n';


    Matrix<double> out{error_tensor.rows(), error_tensor.cols()};
    for (size_t i = 0; i < out.rows(); ++i)
    {
        for (size_t j = 0; j < out.cols(); ++j)
        {
            out(i, j) = m_soft_max_result(i, j) * (error_tensor(i, j) - error_prediction_sum(i));
        }
    }
    return out;
}
