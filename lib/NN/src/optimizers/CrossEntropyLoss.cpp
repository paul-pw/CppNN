#include "CrossEntropyLoss.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>

#include "helpers.hpp"

double CrossEntropyLoss::forward(const Matrix<double> &prediction_tensor,
                                 const Matrix<double> &label_tensor)
{
    assert(prediction_tensor.rows() == label_tensor.rows());
    assert(prediction_tensor.cols() == label_tensor.cols());
    m_prediction_tensor = prediction_tensor;

    double value = 0;
    for (size_t i = 0; i < prediction_tensor.rows(); ++i)
    {
        for (size_t j = 0; j < prediction_tensor.cols(); ++j)
        {
            auto log_val =
                -std::log(prediction_tensor(i, j) + std::numeric_limits<double>::epsilon());
            value += label_tensor(i, j) * log_val;
        }
    }
    return value;
}

Matrix<double> CrossEntropyLoss::backward(const Matrix<double> &label_tensor)
{
    return map(label_tensor, m_prediction_tensor,
               [](auto a, auto b) { return -a / (b + std::numeric_limits<double>::epsilon()); });
}
