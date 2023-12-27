#include <NN.hpp>
#include <cmath>
#include <ranges>
#include <stdexcept>

NN::NN(std::vector<std::unique_ptr<BaseLayer>> layers, std::unique_ptr<Optimizer> optimizer)
    : m_optimizer(std::move(optimizer)), m_layers(std::move(layers))
{
    for (auto &layer : m_layers)
    {
        layer->set_optimizer(m_optimizer);
    }
}

// trains on images and labels
double NN::train(const Matrix<double> &input_batch, const Matrix<double> &label_batch)
{
    // forward
    auto forward_batch = input_batch;
    for (auto &layer : m_layers)
    {
        //std::cout<<forward_batch.tensor()<<'\n';
        forward_batch = layer->forward(forward_batch);
    }
    auto loss = m_loss.forward(forward_batch, label_batch);

    // backward
    auto error_batch = m_loss.backward(label_batch);
    std::ranges::reverse_view layers_reversed{m_layers};
    for (auto &layer : layers_reversed)
    {
        //std::cout<<error_batch.tensor()<<'\n';
        error_batch = layer->backward(error_batch);
    }

    return loss;
}

// evaluates images and returns predictions (only forward pass)
Matrix<double> NN::predict(Matrix<double> input_batch)
{
   // forward
    auto forward_batch = std::move(input_batch);
    for (auto &layer : m_layers)
    {
        //std::cout<<forward_batch.tensor()<<'\n';
        forward_batch = layer->forward(forward_batch);
    }
    return forward_batch;
}
