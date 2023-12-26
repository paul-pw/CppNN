#include <gtest/gtest.h>

#include <helpers.hpp>
#include <layers/FullyConnected.hpp>
#include <layers/ReLU.hpp>
#include <layers/SoftMax.hpp>
#include <limits>
#include <matvec.hpp>
#include <memory>
#include <optimizers/CrossEntropyLoss.hpp>
#include <optimizers/Sgd.hpp>
#include <random>

#include "NN.hpp"

TEST(TestSuit, TestName)
{
    EXPECT_EQ(1, 1);
}

TEST(FullyConnected, forwardShape)
{
    std::mt19937 gen(42);
    Matrix<double> m{5, 12, 2};
    FullyConnected l(12, 3, gen);

    // std::cout<<m.tensor()<<'\n';
    auto out = l.forward(m);
    // std::cout <<out.tensor();

    EXPECT_EQ(out.rows(), 5);
    EXPECT_EQ(out.cols(), 3);
}

TEST(FullyConnected, backwardShape)
{
    std::mt19937 gen(42);
    Matrix<double> m{5, 12, 2};
    Matrix<double> e{5, 3, 2};
    FullyConnected l(12, 3, gen);

    // std::cout<<m.tensor()<<'\n';
    auto f = l.forward(m);
    auto out = l.backward(e);
    // std::cout <<out.tensor();

    EXPECT_EQ(out.rows(), 5);
    EXPECT_EQ(out.cols(), 12);
}

TEST(ReLU, reluShape)
{
    Matrix<double> m{3, 2, 1};
    Matrix<double> e{3, 2, 2};
    m(1, 1) = -1;
    ReLU relu{};
    auto f = relu.forward(m);
    auto b = relu.backward(e);
    // std::cout<<f.tensor()<<'\n' <<b.tensor();
    EXPECT_EQ(f.rows(), 3);
    EXPECT_EQ(f.cols(), 2);
    EXPECT_EQ(b.rows(), 3);
    EXPECT_EQ(b.cols(), 2);
}

TEST(ReLU, relu)
{
    Matrix<double> m{3, 2, 1};
    Matrix<double> e{3, 2, 2};
    m(1, 1) = -1;
    ReLU relu{};
    auto f = relu.forward(m);
    auto b = relu.backward(e);
    // std::cout<<f.tensor()<<'\n' <<b.tensor();
    EXPECT_EQ(f(0, 0), 1);
    EXPECT_EQ(f(1, 1), 0);
    EXPECT_EQ(b(0, 0), 2);
    EXPECT_EQ(b(1, 1), 0);
}

TEST(SoftMax, softmax)
{
    Matrix<double> m{3, 2, 1};
    Matrix<double> e{3, 2, 2};
    m(1, 1) = -1;
    e(1, 1) = 0;
    SoftMax softmax{};
    auto f = softmax.forward(m);
    auto b = softmax.backward(e);
    // std::cout<<f.tensor()<<'\n' <<b.tensor();
    EXPECT_EQ(f(0, 0), 0.5);
    EXPECT_EQ(f(1, 1), 0.11920292202211756);
    EXPECT_EQ(f(1, 0), 0.88079707797788243);
    EXPECT_EQ(b(0, 0), 0);
    EXPECT_EQ(b(1, 1), -0.20998717080701304);
    EXPECT_EQ(b(1, 0), 0.20998717080701307);
}

TEST(CrossEntropy, CrossEntropy)
{
    Matrix<double> m{2, 2};
    m(0, 0) = 0.4;
    m(0, 1) = 0.6;
    m(1, 0) = 0.9;
    m(1, 1) = 0.1;
    Matrix<double> l{2, 2, 1.0};
    l(0, 1) = 0.0;
    l(1, 1) = 0.0;
    CrossEntropyLoss ce{};
    auto loss = ce.forward(m, l);
    // std::cout<<loss;
    EXPECT_DOUBLE_EQ(loss, 1.0216512475319806);
    auto err = ce.backward(l);
    // std::cout <<err.tensor();
    EXPECT_DOUBLE_EQ(err(0, 0), -2.5);
    EXPECT_DOUBLE_EQ(err(0, 1), 0);
    EXPECT_DOUBLE_EQ(err(1, 0), -1.1111111111111107);
    EXPECT_DOUBLE_EQ(err(1, 1), 0);
}

TEST(Sgd, Sgd)
{
    Matrix<double> m{2, 2, 3.0};
    Matrix<double> g{2, 2, 0.4};
    Vector<double> v{2, 2.0};
    Vector<double> gv{2, -0.6};
    Sgd sgd{0.01};
    sgd.update(m, g);
    sgd.update(v, gv);
    EXPECT_DOUBLE_EQ(m(0, 0), 2.996);
    EXPECT_DOUBLE_EQ(v(0), 2.006);
}

TEST(NN, NnTrains)
{
    Matrix<double> input{1, 5, 1.0};
    Matrix<double> labels{1, 3};
    labels(0, 1) = 1;

    std::mt19937 gen(42);
    std::vector<std::unique_ptr<BaseLayer>> layers;
    layers.push_back(std::make_unique<FullyConnected>(5,120,gen));
    layers.push_back(std::make_unique<ReLU>());
    layers.push_back(std::make_unique<FullyConnected>(120,3,gen));
    layers.push_back(std::make_unique<SoftMax>());

    /*{
        std::make_unique<FullyConnected>(5, 120, gen), std::make_unique<ReLU>(),
        std::make_unique<FullyConnected>(120, 3), std::make_unique<SoftMax>()};*/

    NN network{std::move(layers), std::make_unique<Sgd>(1e-2)};

    double loss = std::numeric_limits<double>::max();
    for (size_t i = 0; i < 100; ++i)
    {
        loss = network.train(input, labels);
        //std::cout << loss<<'\n';
    }
    EXPECT_LT(loss, 1e-2);
}
