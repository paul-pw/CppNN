#include <NN.hpp>
#include <cstddef>
#include <helpers.hpp>
#include <iostream>
#include <layers/FullyConnected.hpp>
#include <layers/ReLU.hpp>
#include <layers/SoftMax.hpp>
#include <matvec.hpp>
#include <optimizers/Sgd.hpp>
#include <random>

#include "io.hpp"

int main()
{
    std::mt19937 gen(42);
    std::vector<std::unique_ptr<BaseLayer>> layers;
    layers.push_back(std::make_unique<FullyConnected>(784, 500, gen));
    layers.push_back(std::make_unique<ReLU>());
    layers.push_back(std::make_unique<FullyConnected>(500, 10, gen));
    layers.push_back(std::make_unique<SoftMax>());

    NN network{std::move(layers), std::make_unique<Sgd>(1e-3)};

    // TRAIN
    std::cout << "loading data\n";
    auto training_images = readidx3_batches("./mnist-datasets/train-images.idx3-ubyte", 100);
    auto training_labels = readidx1_batches("./mnist-datasets/train-labels.idx1-ubyte", 100);

    auto epochs = 5;
    std::cout << "training_model for " << epochs << " epochs\n";
    for (size_t i = 0; i < epochs; ++i)
    {
        auto images = training_images[i % training_images.size()];
        auto s = images.shape();
        images.reshape({s[0], s[1] * s[2]});
        Matrix<double> input{std::move(images)};
        auto labels = training_labels[i % training_labels.size()];
        Matrix<double> labels_input{std::move(labels)};

        auto loss = network.train(input, labels_input);
        std::cout << "epoch: " << i << " loss: " << loss << '\n';
    }

    // TEST
    std::cout << "loading test data\n";
    auto testing_images = readidx3_batches("./mnist-datasets/t10k-images.idx3-ubyte", 100);
    auto testing_labels = readidx1_batches("./mnist-datasets/t10k-labels.idx1-ubyte", 100);
    for (size_t i = 0; i < testing_images.size(); ++i)
    {

        auto images = testing_images[i];
        auto s = images.shape();
        images.reshape({s[0], s[1] * s[2]});
        Matrix<double> input{std::move(images)};

        auto labels = testing_labels[i];

        std::cout << "Prediction batch:" << i << '\n';
        auto predicted = network.predict(input);
        for (std::size_t j = 0; j < predicted.rows(); ++j)
        {
            double predicted_max = 0;
            size_t predicted_max_index = 0;
            double label_max = 0;
            size_t label_max_index = 0;

            for (size_t k = 0; k < predicted.cols(); ++k)
            {
                if (predicted(j, k) >= predicted_max)
                {
                    predicted_max = predicted(j, k);
                    predicted_max_index = k;
                }
                if (labels({j, k}) >= label_max)
                {
                    label_max = labels({j, k});
                    label_max_index = k;
                }
            }

            std::cout << " - predicted: " << predicted_max_index << "\tactual: " << label_max_index
                      << '\n';
        }
    }
}
