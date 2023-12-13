#include <cstddef>
#include "BaseLayer.h"

class FullyConnected : BaseLayer{
    FullyConnected(std::size_t input_size, std::size_t output_size);
    Tensor<double> forward(Tensor<double> input_tensor) override;
    Tensor<double> backward(Tensor<double> error_tensor) override;
    ~FullyConnected() override = default;
};
