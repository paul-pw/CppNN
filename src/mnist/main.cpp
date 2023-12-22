#include <iostream>
#include <matvec.hpp>
#include <helpers.hpp>
#include <random>

int main(){
    std::cout << "hello Mnist\n";
    std::mt19937 gen(43);
    auto t = random_tensor({2,4}, 0.0, 1.0, gen);
    std::cout << t;
}
