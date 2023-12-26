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
    printf("Test1\n");
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

/*
1.4 Batch Normalization
Batch Normalization is a regularization technique which is conceptually very well known in
Machine Learning but specially adapted to Deep Learning.
Task:
Implement a class BatchNormalization in the file "BatchNormalization.py" in folder "Lay-
ers". This class has to provide the methods forward(input_tensor) and
backward(error_tensor).

 -Implement the constructor for this layer which receives the argument channels. chan-
nels denotes the number of channels of the input tensor in both, the vector and the
image-case. Initialize the bias \beta and the weights \gamma
 according to the channels-size using the method initialize. This layer has trainable parameters, so remember to set the
inherited member trainable accordingly.

 -Implement the method initialize which initializes the weights \gamma
 and the biases \beta. initialize ignores any assigned initializer and initializes always the weights \gamma
 with ones and the biases \beta with zeros, since you do not want the weights \gamma
 and bias \beta to have an impact
at the beginning of the training. Make sure you optimize the weights and bias in the
backward pass, but only if optimizers are defined.

 -Implement the Batch Normalization methods forward(input_tensor) and
backward(error_tensor) with independent activations for the training phase. Make
sure you use an \epsilon smaller than 1e-10.

 -Hint: In Helpers.py we provide a function compute_bn_gradients(error_tensor, input_tensor, weights, mean, var) for the computation of the gradient w.r.t. inputs.
Note that this function does not compute the gradient w.r.t. the weights.

 -Implement the moving average estimation of training set mean and variance.

 -Modify the Batch Normalization method forward(input_tensor) for the testing phase.
Use an online estimation of the mean and variance. Initialize mean and variance with
the batch mean and the batch standard deviation of the first batch used for training.

 -Implement the convolutional variant. The layer should change behaviour depending on
the shape of the input_tensor.

 -Implement a method reformat(tensor) which receives the tensor that must be reshaped.
Depending on the shape of the tensor, the method reformats the image-like tensor
 (with 4 dimension) into its vector-like variant (with 2 dimensions), and the same method
 reformats the vector-like tensor into its image-like tensor variant. Use this in the forward
and the backward pass.



 */
