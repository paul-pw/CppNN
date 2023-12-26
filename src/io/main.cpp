#include <iostream>
#include <filesystem>
#include "tensor.hpp"
#include "io.cpp"

int main(int argc, char* argv[]) {
     if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <label_input_path> <output_path> <label_index>\n";
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    size_t image_index = std::stoul(argv[3]);
    //size_t image_index = 0;

   /* std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;
    std::string input_path = "../../mnist-datasets/train-images.idx3-ubyte";
    std::string output_path = "../../mnist-datasets/image_out.txt";
    std::string label_path = "../../mnist-datasets/train-labels.idx1-ubyte";
    std::string label_output_path = "../../mnist-datasets/label_out.txt";
    size_t image_index = 0;
    size_t label_index = 0; // Change to read a specific label index*/

    try {
        auto tensor = readidx3(input_path, image_index);
        writeTensorToFile(tensor, output_path);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

   /* try {
        auto tensor = readidx1(label_path, label_index); // Read label and one-hot encode
        writeTensorToFile(tensor, label_output_path); // Write the one-hot encoded tensor to file
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Done.\n";
    return 0;*/
}
