#include <helpers.hpp>
#include <iostream>
#include <layers/FullyConnected.hpp>
#include <matvec.hpp>
#include <random>

int main()
{
    std::cout << "hello Mnist\n";
    std::mt19937 gen(43);
    auto t = random_tensor({2, 4}, 0.0, 1.0, gen);
    FullyConnected l(12, 54, gen);
    std::cout << t;
}
