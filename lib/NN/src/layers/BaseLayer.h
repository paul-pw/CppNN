#include <tensor.hpp>

class BaseLayer
{
    virtual Tensor<double> forward(Tensor<double> input_tensor) = 0;
    virtual Tensor<double> backward(Tensor<double> error_tensor) = 0;
    virtual ~BaseLayer() = 0;
};
