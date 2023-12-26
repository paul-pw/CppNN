#pragma once

#include "optimizers/Optimizer.hpp"

class Sgd : public Optimizer
{
public:
    explicit Sgd(double learning_rate);
    void update(Matrix<double> &weight_tensor, const Matrix<double> &gradient_tensor) override;
    void update(Vector<double> &bias_tensor, const Vector<double> &gradient_tensor) override;
    std::unique_ptr<Optimizer> clone() override;

private:
    double m_learning_rate;
};
