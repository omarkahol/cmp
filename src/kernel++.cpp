#include "kernel++.h"

std::shared_ptr<cmp::kernel::Kernel> cmp::kernel::operator+(std::shared_ptr<Kernel> k1, std::shared_ptr<Kernel> k2)
{
    return cmp::kernel::Sum::make(k1, k2);
}

std::shared_ptr<cmp::kernel::Kernel> cmp::kernel::operator*(std::shared_ptr<Kernel> k1, std::shared_ptr<Kernel> k2)
{
    return cmp::kernel::Product::make(k1, k2);
}
