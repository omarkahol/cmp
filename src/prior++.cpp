#include "prior++.h"

std::shared_ptr<cmp::prior::Prior> cmp::prior::operator*(std::shared_ptr<Prior> p1, std::shared_ptr<Prior> p2)
{
    return cmp::prior::Product::make(p1, p2);
}