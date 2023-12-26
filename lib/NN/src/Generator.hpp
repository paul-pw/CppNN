#pragma once
#include <matvec.hpp>

class Generator
{
public:
    // TODO use batch size in child constructor
    struct Data
    {
        Matrix<double> data_batch;
        Matrix<double> label_batch;
    };
    virtual Data next() = 0;
    virtual ~Generator() = default;
};
