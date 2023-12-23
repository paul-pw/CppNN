#include <gtest/gtest.h>
#include <layers/FullyConnected.hpp>
#include <matvec.hpp>
#include <helpers.hpp>
#include <random>


TEST(TestSuit, TestName)
{
    EXPECT_EQ(1, 1);
}

TEST(FullyConnected, forwardShape){
    std::mt19937 gen(42);
    Matrix<double> m{5, 12, 2};
    FullyConnected l(12, 3, gen);
    
    //std::cout<<m.tensor()<<'\n';
    auto out = l.forward(m);
    //std::cout <<out.tensor();

    EXPECT_EQ(out.rows(), 5);
    EXPECT_EQ(out.cols(), 3);

}
