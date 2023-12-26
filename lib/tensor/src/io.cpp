// Description: MNIST data read function
#include "io.hpp"
#include <iostream>
#include <fstream>
#include "tensor.hpp"


// Helper function to read a 32-bit big endian integer
int readInt32BE(std::ifstream &file) {
    char bytes[4];
    file.read(bytes, 4);
    return ((bytes[0] & 0xFF) << 24) |
           ((bytes[1] & 0xFF) << 16) |
           ((bytes[2] & 0xFF) << 8) |
           (bytes[3] & 0xFF);
}

// Function to read MNIST label data
Tensor<double> readidx3(std::filesystem::path path, size_t image_index) {

    // Open the MNIST image file
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + path.string());
    }

    // Read the magic number for images(0x00000803)
    int magic_number = readInt32BE(file);
    if (magic_number != 0x00000803) {
        throw std::runtime_error("Invalid magic number in file " + path.string());
    }

    // Read the dimensions
    int num_images = readInt32BE(file);
    int num_rows = readInt32BE(file);
    int num_columns = readInt32BE(file);

    // Skip to the image at the specified index
    size_t image_size = num_rows * num_columns;
    file.seekg(image_index * image_size, std::ios_base::cur);

    // Prepare the tensor to hold all images
    Tensor<double> image({static_cast<size_t>(num_rows), static_cast<size_t>(num_columns)});

    // Read the image and normalize pixel values
    for (int r = 0; r < num_rows; ++r) {
        for (int c = 0; c < num_columns; ++c) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            image({static_cast<size_t>(r), static_cast<size_t>(c)}) = static_cast<double>(pixel) / 255.0;
        }
    }

    if (!file) {
        throw std::runtime_error("Error reading file " + path.string());
    }

    return image;
}

// Function to read MNIST Image data
Tensor<double> readidx1(std::filesystem::path path, size_t label_index) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + path.string());
    }

    // Read and check the magic number for label(0x00000801)
    int magic_number = readInt32BE(file);
    if (magic_number != 0x00000801) {
        throw std::runtime_error("Invalid magic number in file " + path.string());
    }

    // Read the number of labels
    int num_labels = readInt32BE(file);

    // Skip to the label at the specified index
    file.seekg(label_index, std::ios_base::cur);

    // Prepare the tensor for a single one-hot encoded label
    Tensor<double> one_hot_label({10}, 0.0);  // Initialize all elements to 0.0

    // Read the label
    unsigned char label = 0;
    file.read(reinterpret_cast<char*>(&label), sizeof(label));

    // One-hot encode the label
    if (label < 10) {
        one_hot_label(static_cast<size_t>(label)) = 1.0;
    } else {
        throw std::runtime_error("Invalid label value read from file");
    }

    if (!file) {
        throw std::runtime_error("Error reading file " + path.string());
    }

    return one_hot_label;
}
