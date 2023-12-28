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
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <vector>
#include <utility>
#include <chrono>
#include <io.hpp>

// Function to trim whitespace from a string
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    size_t last = str.find_last_not_of(' ');
    if (first == std::string::npos || last == std::string::npos) {
        return "";
    }
    return str.substr(first, (last - first + 1));
}

// Function to parse the configuration file
std::unordered_map<std::string, std::string> parseConfig(const std::filesystem::path& filepath) {
    std::unordered_map<std::string, std::string> config;
    std::ifstream file(filepath);
    if (!file) {
        std::cerr << "Error opening config file: " << filepath << std::endl;
        return config;
    }

    std::string line;
    while (std::getline(file, line)) {
        auto comment_pos = line.find("//");
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos); // Strip comments
        }
        line = trim(line); // Trim the line after removing comments
        if (line.empty()) continue; // Skip empty lines

        std::istringstream is_line(line);
        std::string key, value;
        if (std::getline(is_line, key, '=') && std::getline(is_line, value)) {
            config[trim(key)] = trim(value);
        }
    }
    return config;
}

static int totalImagesProcessed = 0;

// Function to log predictions to a file in the specified format
void logPredictions(const std::filesystem::path& logFilePath,
                    const std::vector<std::pair<int, int>>& predictions,
                    int batchSize) {
    std::ofstream logFile(logFilePath, std::ios::app); // Open in append mode to add to existing logs
    if (!logFile) {
        std::cerr << "Error opening log file: " << logFilePath << std::endl;
        return;
    }

    // Calculate the current batch number based on the total number of images processed so far
    int currentBatch = totalImagesProcessed / batchSize;

    // Log the batch header
    logFile << "Current batch: " << currentBatch << std::endl;

    for (const auto& prediction : predictions) {
        // Log the prediction with the current image index
        logFile << "- image " << totalImagesProcessed << ": Prediction=" << prediction.first;
        logFile << ". Label=" << prediction.second << std::endl;
        totalImagesProcessed++; // Increment the total image count
    }

    logFile.close();
}

// Reset the count when starting a new complete run (e.g., at the beginning of the main function)
void resetBatchCount() {
    totalImagesProcessed = 0;
}



int main()
{

    // Obtain the project root path from the current file's path (assuming this file is located in the src directory)
    std::filesystem::path currentPath = std::filesystem::current_path();
    std::filesystem::path projectRoot = currentPath.parent_path().parent_path();
    std::filesystem::path filepath = projectRoot / "mnist-configs" / "input-ci.config";
   // std::filesystem::path filepath_out = projectRoot / "mnist-configs" / "output.config";

    std::cout << "Config file path: " << filepath << std::endl;
    // Check if the file exists
    if (!std::filesystem::exists(filepath)) {
        std::cerr << "Config file does not exist at the path: " << filepath << std::endl;
        return 1;
    }

    // Check file permissions (readable)
    std::ifstream configFile(filepath);
    if (!configFile.good()) {
        std::cerr << "Cannot open config file, please check the file permissions." << std::endl;
        return 1;
    }
    configFile.close();



    auto config = parseConfig(filepath);
    if (config.empty()) {
        std::cerr << "Failed to parse config file." << std::endl;
        return 1;
    }


    resetBatchCount(); // Reset the batch count at the beginning of a new run
    int batchSize = 1; // Default batch size if not specified in the config
    if (config.find("batch_size") != config.end()) {
        batchSize = std::stoi(config["batch_size"]);
    }

    // Determine the log file path from the configuration
    std::filesystem::path logFilePath = projectRoot / "mnist-configs" / config["rel_path_log_file"];

    // Open the log file to write
    std::ofstream logFile(logFilePath, std::ios::app); // Open in append mode to add to existing logs
    if (!logFile.is_open()) {
        std::cerr << "Failed to open or create log file: " << logFilePath << std::endl;
        return 1;
    }


    logFile.close();

    // ... Code to set up and train your neural network goes here ...

    // Code to log predictions - replace with actual predictions
    std::vector<std::pair<int, int>> predictions = {{1, 1}, {1, 1}, {1,1 }};// Replace with actual data
    logPredictions(logFilePath, predictions,batchSize); // Pass the correct path variable

    std::cout << "Done" << std::endl;
    return 0;


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

    auto epochs = 100;
    std::cout << "training_model for " << epochs << " epochs\n";
    for (size_t i = 0; i < epochs; ++i)
    {
        auto images = training_images[i % training_images.size()];
        auto s = images.shape();
        images.reshape({s[0], s[1] * s[2]});
        Matrix<double> input{std::move(images)};
        auto labels = training_labels[i % training_labels.size()];
        Matrix<double> labels_input{std::move(labels)};
        
        auto start = std::chrono::high_resolution_clock::now();
        auto loss = network.train(input, labels_input);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = duration_cast<std::chrono::milliseconds>(stop - start); 
        std::cout <<"duration: "<< duration.count() <<"ms\n";
        
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
        double accurary = 0.0;
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
            if(predicted_max_index == label_max_index){
                accurary += 1.0/100; // Batch Size
            }
        }
        std::cout << "accuracy: "<<accurary <<'\n';
    }
}
