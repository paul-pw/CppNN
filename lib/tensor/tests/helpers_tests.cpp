#include <gtest/gtest.h>

#include <helpers.hpp>
#include <iostream>
#include <matvec.hpp>

TEST(TestSuitName, SingleTestName)
{
    EXPECT_EQ(1, 1);
}

TEST(helpers, randomTensor)
{
    std::mt19937 gen(43);
    auto t = random_tensor({2, 4}, 0.0, 1.0, gen);
    auto s = t.shape();
    EXPECT_EQ(s[0], 2);
    EXPECT_EQ(s[1], 4);
    EXPECT_EQ(s.size(), 2);
    EXPECT_DOUBLE_EQ(t({0, 0}), 0.49686129393872214);
}

TEST(helpers, transpose)
{
    Matrix<int> t{4, 6, 0};
    t(3, 2) = 1;
    auto t1 = transpose(t);
    EXPECT_EQ(t1.rows(), t.cols());
    EXPECT_EQ(t1.cols(), t.rows());
    EXPECT_EQ(t1(2, 3), 1);
}

TEST(helpers, dot)
{
    Matrix<int> a{2, 3, 2};
    Matrix<int> b{3, 5, 3};
    a(1, 1) = 5;
    b(1, 2) = 4;
    auto c = dot(a, b);
    //std::cout << a.tensor() << '\n' << b.tensor() << '\n' << c.tensor();
    EXPECT_EQ(c(0, 0), 18);
    EXPECT_EQ(c(0, 2), 20);
    EXPECT_EQ(c(1, 0), 27);
    EXPECT_EQ(c(1, 2), 32);
}

TEST(helpers, add){
    Matrix<int> d{2,3,1};
    Matrix<int> e{2,3,1};
    Vector<int> b{2,1};
    b(1) = 2;
    Vector<int> c{3,2};
    c(1) = 3;
    add(d,b, Axis::row);
    add(e,c, Axis::col);
    //std::cout << c.tensor() <<'\n' <<e.tensor();
    EXPECT_EQ(d(0,0), 2);
    EXPECT_EQ(d(1,0), 3);
    EXPECT_EQ(e(0,0), 3);
    EXPECT_EQ(e(0,1), 4);
}

TEST(helpers, sumAxis){
    Matrix<int> m{2,3,2};
    m(1,1) = 1;
    auto a = sum_axis(m, Axis::col);
    auto b = sum_axis(m, Axis::row);
    //std::cout << m.tensor()<<'\n'<<a.tensor() << '\n' << b.tensor();
    EXPECT_EQ(a.size(), 3);
    EXPECT_EQ(a(0), 4);
    EXPECT_EQ(a(1), 3);
    EXPECT_EQ(b.size(), 2);
    EXPECT_EQ(b(0), 6);
    EXPECT_EQ(b(1), 5);
}

TEST(helpers, map){
    Matrix<int> m1{2,3,2};
    Matrix<int> m2{2,3,3};
    auto o1 = map(m1, [](auto i){return i*2;});
    auto o2 = map(m1, m2, [](auto a, auto b){return a*b;});
    EXPECT_EQ(o1.rows(), 2);
    EXPECT_EQ(o2.rows(), 2);
    EXPECT_EQ(o1(0,0), 4);
    EXPECT_EQ(o2(0,0), 6);
}
