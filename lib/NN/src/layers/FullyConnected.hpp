#pragma once

#include <random>

#include "../optimizers/Optimizer.hpp"
#include "BaseLayer.hpp"
#include "matvec.hpp"
#include "tensor.hpp"

class FullyConnected : BaseLayer
{
public:
    FullyConnected(std::size_t input_size, std::size_t output_size, std::mt19937 &generator);
    Matrix<double> forward(const Matrix<double> &input_tensor) override;
    Matrix<double> backward(const Matrix<double> &error_tensor) override;
    void set_optimizer(const std::unique_ptr<Optimizer> &opt) override;
    //~FullyConnected() override = default;

private:
    Matrix<double> m_weights;
    Vector<double> m_bias;
    std::unique_ptr<Optimizer> m_weight_optimizer;
    std::unique_ptr<Optimizer> m_bias_optimizer;
    Matrix<double> m_input_tensor;
};
