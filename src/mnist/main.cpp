#include <NN.hpp>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <helpers.hpp>
#include <io.hpp>
#include <iostream>
#include <layers/FullyConnected.hpp>
#include <layers/ReLU.hpp>
#include <layers/SoftMax.hpp>
#include <matvec.hpp>
#include <optimizers/Sgd.hpp>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Function to trim whitespace from a string
std::string trim(const std::string &str)
{
    size_t first = str.find_first_not_of(' ');
    size_t last = str.find_last_not_of(' ');
    if (first == std::string::npos || last == std::string::npos)
    {
        return "";
    }
    return str.substr(first, (last - first + 1));
}

// Function to parse the configuration file
std::unordered_map<std::string, std::string> parseConfig(const std::filesystem::path &filepath)
{
    std::unordered_map<std::string, std::string> config;

    // TODO START Remove this part later
    std::cout << "config Dump\n";
    std::ifstream f(filepath);
    if (f.is_open())
        std::cout << f.rdbuf();
    std::cout << "config Dump End\n\n";
    // TODO END remvoe this part later

    std::ifstream file(filepath);
    if (!file)
    {
        std::cerr << "Error opening config file: " << filepath << std::endl;
        return config;
    }

    std::string line;
    while (std::getline(file, line))
    {
        auto comment_pos = line.find("//");
        if (comment_pos != std::string::npos)
        {
            line = line.substr(0, comment_pos); // Strip comments
        }
        line = trim(line); // Trim the line after removing comments
        if (line.empty())
            continue; // Skip empty lines

        std::istringstream is_line(line);
        std::string key, value;
        if (std::getline(is_line, key, '=') && std::getline(is_line, value))
        {
            config[trim(key)] = trim(value);
        }
    }
    return config;
}

// Function to log predictions to a file in the specified format
void logPredictions(const std::filesystem::path &logFilePath,
                    const std::vector<std::pair<size_t, size_t>> &predictions, size_t batchSize,
                    size_t currentBatch)
{
    std::ofstream logFile(logFilePath,
                          std::ios::app); // Open in append mode to add to existing logs
    if (!logFile)
    {
        std::cerr << "Error opening log file: " << logFilePath << std::endl;
        return;
    }

    // Calculate the current batch number based on the total number of images processed so far
    size_t totalImagesProcessed = currentBatch * batchSize;

    // Log the batch header
    logFile << "Current batch: " << currentBatch << std::endl;

    for (const auto &prediction : predictions)
    {
        // Log the prediction with the current image index
        logFile << "- image " << totalImagesProcessed << ": Prediction=" << prediction.first;
        logFile << ". Label=" << prediction.second << std::endl;
        totalImagesProcessed++; // Increment the total image count
    }

    logFile.close();
}

int main(int argc, char *argv[])
{

    // Obtain the project root path from the current file's path (assuming this file is located in
    // the src directory)
    std::filesystem::path currentPath = std::filesystem::current_path();
    std::filesystem::path filepath = currentPath;
    if (argc > 1)
    {
        filepath /= argv[1];
    }
    else
    {
        filepath = filepath / "mnist-configs" / "input.config";
    }

    std::cout << "Config file path: " << filepath << std::endl;
    // Check if the file exists
    if (!std::filesystem::exists(filepath))
    {
        std::cerr << "Config file does not exist at the path: " << filepath << std::endl;
        return 1;
    }

    // Check file permissions (readable)
    std::ifstream configFile(filepath);
    if (!configFile.good())
    {
        std::cerr << "Cannot open config file, please check the file permissions." << std::endl;
        return 1;
    }
    configFile.close();

    auto config = parseConfig(filepath);
    if (config.empty())
    {
        std::cerr << "Failed to parse config file.\n";
        return 1;
    }

    size_t epochs = std::stoi(config["num_epochs"]);
    size_t batch_size = std::stoi(config["batch_size"]);
    size_t hidden_size = std::stoi(config["hidden_size"]);
    double learning_rate = std::stod(config["learning_rate"]);
    std::filesystem::path rel_path_train_images = config["rel_path_train_images"];
    std::filesystem::path rel_path_train_labels = config["rel_path_train_labels"];
    std::filesystem::path rel_path_test_images = config["rel_path_test_images"];
    std::filesystem::path rel_path_test_labels = config["rel_path_test_labels"];
    std::filesystem::path rel_path_log_file = config["rel_path_log_file"];

    std::cout << "epochs: " << epochs << "\nbatch_size: " << batch_size
              << "\nhidden_size: " << hidden_size << "\nlearning_rate: " << learning_rate
              << "\ntrain_images: " << rel_path_train_images
              << "\ntrain_labels: " << rel_path_train_labels
              << "\ntest_images: " << rel_path_test_images << "\ntest_labels"
              << rel_path_test_labels << "\nlog_file: " << rel_path_log_file << '\n';

    // Determine the log file path from the configuration
    std::filesystem::path logFilePath = currentPath / config["rel_path_log_file"];

    // Set up the Network
    std::mt19937 gen(42);
    std::vector<std::unique_ptr<BaseLayer>> layers;
    layers.push_back(std::make_unique<FullyConnected>(784, hidden_size, gen));
    layers.push_back(std::make_unique<ReLU>());
    layers.push_back(std::make_unique<FullyConnected>(hidden_size, 10, gen));
    layers.push_back(std::make_unique<SoftMax>());

    NN network{std::move(layers), std::make_unique<Sgd>(learning_rate)};

    // TRAIN
    std::cout << "\nloading data\n";
    auto training_images = readidx3_batches(rel_path_train_images, batch_size);
    auto training_labels = readidx1_batches(rel_path_train_labels, batch_size);

    std::cout << "\ntraining model\n";
    for (size_t i = 0; i < epochs; ++i)
    {
        std::cout << "\n##########\n EPOCH: " << i+1 << "\n##########\n";
        for(size_t j = 0; j<training_images.size(); ++j){
            // Prepare images and labels for training
            auto images = training_images[j];
            auto labels = training_labels[j];

            auto s = images.shape();
            images.reshape({s[0], s[1] * s[2]});
            Matrix<double> input{std::move(images)};
            Matrix<double> labels_input{std::move(labels)};

            // Train model (and time training)
            auto start = std::chrono::high_resolution_clock::now();
            auto loss = network.train(input, labels_input);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = duration_cast<std::chrono::milliseconds>(stop - start);

            std::cout << "batch: " << j << " loss: " << loss << " duration: " << duration.count()
                      << "ms" << '\n';
        }
    }

    // TEST
    std::cout << "\nloading test data\n";
    auto testing_images = readidx3_batches(rel_path_test_images, batch_size);
    auto testing_labels = readidx1_batches(rel_path_test_labels, batch_size);

    double cumulated_accurary = 0.0;
    for (size_t i = 0; i < testing_images.size(); ++i)
    {
        // prepare images ad data
        auto images = testing_images[i];
        auto s = images.shape();
        images.reshape({s[0], s[1] * s[2]});
        Matrix<double> input{std::move(images)};
        auto labels = testing_labels[i];

        auto predicted = network.predict(input);
        double accurary = 0.0;
        std::vector<std::pair<size_t, size_t>> predictions{};
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

            predictions.emplace_back(predicted_max_index, label_max_index);
            if (predicted_max_index == label_max_index)
            {
                accurary += 1.0 / static_cast<double>(batch_size); // Batch Size
            }
        }
        cumulated_accurary += accurary;
        std::cout << "Prediction batch:" << i << " accuracy: " << accurary
                  << " cumulated accuracy: " << cumulated_accurary / (1.0 + i) << '\n';
        logPredictions(rel_path_log_file, predictions, batch_size, i);
    }
}
