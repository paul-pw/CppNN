#pragma once

#include <memory>
#include <vector>
#include "layers/BaseLayer.hpp"
#include "matvec.hpp"
#include "optimizers/CrossEntropyLoss.hpp"
#include "optimizers/Optimizer.hpp"
#include "Generator.hpp"

class NN{
public:
    NN(std::vector<std::unique_ptr<BaseLayer>> layers, std::unique_ptr<Optimizer> optimizer);
    
    // TODO train with generator
    void train(std::unique_ptr<Generator> gen, size_t epochs);

    // trains on images and labels and returns loss
    double train(const Matrix<double>& input_batch, const Matrix<double>& label_batch);

    // evaluates images and returns predictions (only forward pass)
    Matrix<double> evaluate(Matrix<double> input_batch);
private:
    std::vector<std::unique_ptr<BaseLayer>> m_layers;
    std::unique_ptr<Optimizer> m_optimizer;
    CrossEntropyLoss m_loss;

};
