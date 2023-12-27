#include <helpers.hpp>
#include <iostream>
#include <layers/FullyConnected.hpp>
#include <matvec.hpp>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <vector>
#include <utility>
/*int main()
{
    std::cout << "hello Mnist\n";
    std::mt19937 gen(43);
    auto t = random_tensor({2, 4}, 0.0, 1.0, gen);
    Matrix<double> m{1, 12};
    FullyConnected l(12, 3, gen);
    std::cout<<l.forward(Matrix<double>{1,12}).tensor()<<'\n';
    std::cout << t;
}*/


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

// Main function
int main() {
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
}