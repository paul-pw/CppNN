#include <iostream>
#include <filesystem>
#include "tensor.hpp"
#include <io.hpp>

// Funktion zum Lesen der Magic Number
int readMagicNumber(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + path);
    }
    return readInt32BE(file);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_path> <output_path> <index>\n";
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    size_t index = std::stoul(argv[3]);

    try {
        int magic_number = readMagicNumber(input_path);

        Tensor<double> tensor;
        if (magic_number == 0x00000803) {
            tensor = readidx3(input_path, index);
        } else if (magic_number == 0x00000801) {
            tensor = readidx1(input_path, index);
        } else {
            throw std::runtime_error("Unknown magic number");
        }

        writeTensorToFile(tensor, output_path);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Done.\n";
    return 0;
}
