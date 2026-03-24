#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <cmath>

namespace Utils {

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}



}

#endif // UTILS_H
