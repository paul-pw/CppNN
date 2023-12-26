#pragma once
#include <matvec.hpp>

class CrossEntropyLoss
{
public:
    double forward(const Matrix<double> &input_tensor, const Matrix<double> &label_tensor);
    Matrix<double> backward(const Matrix<double> &label_tensor);

private:
    Matrix<double> m_prediction_tensor;
};
