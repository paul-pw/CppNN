#include <NN.hpp>
#include <ranges>

NN::NN(std::vector<std::unique_ptr<BaseLayer>> layers, std::unique_ptr<Optimizer> optimizer)
    : m_optimizer(std::move(optimizer)), m_layers(std::move(layers))
{
    for(auto& layer : m_layers){
        layer->set_optimizer(m_optimizer);
    }
}

// TODO train with generator
void NN::train(std::unique_ptr<Generator> gen, size_t epochs)
{
}

// trains on images and labels
double NN::train(const Matrix<double>& input_batch, const Matrix<double>& label_batch)
{
    // forward
    auto forward_batch = input_batch;
    for (auto& layer : m_layers)
    {
        forward_batch = layer->forward(forward_batch);
    }
    auto loss = m_loss.forward(forward_batch, label_batch);

    // backward
    auto error_batch = m_loss.backward(label_batch);
    std::ranges::reverse_view layers_reversed{m_layers};
    for (auto& layer : layers_reversed)
    {
        error_batch = layer->backward(error_batch);
    }

    return loss;
}

// evaluates images and returns predictions (only forward pass)
Matrix<double> NN::evaluate(Matrix<double> input_batch)
{
}
