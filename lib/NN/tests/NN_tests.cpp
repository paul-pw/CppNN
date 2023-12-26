#include <gtest/gtest.h>
#include <layers/FullyConnected.hpp>
#include <layers/ReLU.hpp>
#include <layers/SoftMax.hpp>
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

TEST(FullyConnected, backwardShape)
{
    std::mt19937 gen(42);
    Matrix<double> m{5, 12, 2};
    Matrix<double> e{5, 3, 2};
    FullyConnected l(12, 3, gen);
    
    //std::cout<<m.tensor()<<'\n';
    auto f = l.forward(m);
    auto out = l.backward(e);
    //std::cout <<out.tensor();

    EXPECT_EQ(out.rows(), 5);
    EXPECT_EQ(out.cols(), 12);
}

TEST(ReLU, reluShape){
    Matrix<double> m{3,2,1};
    Matrix<double> e{3,2,2};
    m(1,1) = -1;
    ReLU relu{};
    auto f = relu.forward(m);
    auto b = relu.backward(e);
    //std::cout<<f.tensor()<<'\n' <<b.tensor();
    EXPECT_EQ(f.rows(), 3);
    EXPECT_EQ(f.cols(), 2);
    EXPECT_EQ(b.rows(), 3);
    EXPECT_EQ(b.cols(), 2);
}

TEST(ReLU, relu){
    Matrix<double> m{3,2,1};
    Matrix<double> e{3,2,2};
    m(1,1) = -1;
    ReLU relu{};
    auto f = relu.forward(m);
    auto b = relu.backward(e);
    //std::cout<<f.tensor()<<'\n' <<b.tensor();
    EXPECT_EQ(f(0,0), 1);
    EXPECT_EQ(f(1,1), 0);
    EXPECT_EQ(b(0,0), 2);
    EXPECT_EQ(b(1,1), 0);
}


TEST(SoftMax, softmax){
    Matrix<double> m{3,2,1};
    Matrix<double> e{3,2,2};
    m(1,1) = -1;
    e(1,1) = 0;
    SoftMax softmax{};
    auto f = softmax.forward(m);
    auto b = softmax.backward(e);
    //std::cout<<f.tensor()<<'\n' <<b.tensor();
    EXPECT_EQ(f(0,0), 0.5);
    EXPECT_EQ(f(1,1), 0.11920292202211756);
    EXPECT_EQ(f(1,0), 0.88079707797788243);
    EXPECT_EQ(b(0,0), 0);
    EXPECT_EQ(b(1,1), -0.20998717080701304);
    EXPECT_EQ(b(1,0), 0.20998717080701307);
}
