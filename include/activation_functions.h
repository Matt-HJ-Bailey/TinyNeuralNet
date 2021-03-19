#pragma once
#include <cmath>

class Sigmoid {
    public:
        double function(const double x) const{ return 1.0/(1.0 + std::exp(-x));};
        double gradient(const double x) const{ return function(x) * (1.0 - function(x)); };
};

class RectifiedLinearUnit {
    public:
        constexpr double function(const double x) const{ return x > 0.0 ? x : 0.0; };
        constexpr double gradient(const double x) const{ return x > 0.0 ? 1 : 0.0; };
};

class Tanh {
    public:
        double function(const double x) const{ return std::tanh(x); };
        double gradient(const double x) const{ return 1 - function(x) * function(x);};
};
