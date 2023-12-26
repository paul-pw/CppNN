#include "Sgd.hpp"

#include <cassert>
#include <memory>

#include "optimizers/Optimizer.hpp"

Sgd::Sgd(double learning_rate) : m_learning_rate(learning_rate)
{
}

void Sgd::update(Matrix<double> &weight_tensor, const Matrix<double> &gradient_tensor)
{
    assert(weight_tensor.rows() == gradient_tensor.rows());
    assert(weight_tensor.cols() == gradient_tensor.cols());
    for (size_t i = 0; i < weight_tensor.rows(); ++i)
    {
        for (size_t j = 0; j < weight_tensor.cols(); ++j)
        {
            weight_tensor(i, j) -= m_learning_rate * gradient_tensor(i, j);
        }
    }
}

void Sgd::update(Vector<double> &bias_tensor, const Vector<double> &gradient_tensor)
{
    assert(bias_tensor.size() == gradient_tensor.size());
    for (size_t i = 0; i < bias_tensor.size(); ++i)
    {
        bias_tensor(i) -= m_learning_rate * gradient_tensor(i);
    }
}

std::unique_ptr<Optimizer> Sgd::clone()
{
    return std::make_unique<Sgd>(m_learning_rate);
}
