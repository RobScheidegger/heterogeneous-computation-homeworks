#pragma once

#include <stdint.h>
#include <utility>
#include <vector>

template <typename T, typename F = float>
std::pair<F, F> getMeanAndStdDev(const std::vector<T>& v) {
    F sum = std::accumulate(v.begin(), v.end(), 0.0);
    F mean = sum / v.size();

    F sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    F stdev = std::sqrt(sq_sum / v.size() - mean * mean);
    return std::make_pair(mean, stdev);
}