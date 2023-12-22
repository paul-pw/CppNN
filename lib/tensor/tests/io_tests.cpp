#include <gtest/gtest.h>
#include <helpers.hpp>
#include <matvec.hpp>

TEST(TestSuitName, SingleTestName){
    EXPECT_EQ(1,1);
}

TEST(helpers, randomTensor){
    std::mt19937 gen(43);
    auto t = random_tensor({2,4}, 0.0, 1.0, gen);
    auto s = t.shape();
    EXPECT_EQ(s[0],2);
    EXPECT_EQ(s[1],4);
    EXPECT_EQ(s.size(), 2);
    EXPECT_DOUBLE_EQ(t({0,0}), 0.49686129393872214);
}

TEST(helpers, transpose)
{
    Matrix<int> t{4,6, 0};
    t(3,2) = 1;
    auto t1 = transpose(t);
    EXPECT_EQ(t1.rows(), t.cols());
    EXPECT_EQ(t1.cols(), t.rows());
    EXPECT_EQ(t1(2,3),1);
}
