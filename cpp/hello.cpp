#include <iostream>
#include <vector>

#include <pybind11/pybind11.h>

int add(int i, int j)
{
    return i + j;
}

PYBIND11_MODULE(example, m)
{
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
}

int sum_exps(int len, int exp)
{
    int res = 0;
    for (int n = 0; n < len; ++n)
    {
        res += std::pow(n, exp)
    }

    return res;
}

int my_add(int len)
{
    std::vector<int> results;
    int res = 1;
    for (int n = 1; n < len; ++n)
    {
        int mod = res % n;
        std::cout << res << " " << mod << " " << n << std::endl;
    }

    return res;
}

int main()
{
    int res = sum_exps(50, 5);
    std::cout << res << std::endl;

    return 0;
}
