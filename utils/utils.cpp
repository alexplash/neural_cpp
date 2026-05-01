#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <random>
#include <algorithm>

std::vector<float> random_vector(int size, float min, float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);

    std::vector<float> vec(size);
    std::generate(vec.begin(), vec.end(), [&]() { return dist(gen); });
    return vec;
}

std::vector<std::vector<float>> random_matrix(int size_i, int size_j, float min, float max) {
    std::vector<std::vector<float>> matrix;

    for (std::size_t i = 0; i < size_i; i++) {
        std::vector<float> row(size_j);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(min, max);

        std::generate(row.begin(), row.end(), [&]() { return dist(gen); });
        matrix.push_back(row);
    }

    return matrix;
}

float random_float(float min, float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);

    std::vector<float> vec(1);
    std::generate(vec.begin(), vec.end(), [&]() { return dist(gen); });
    return vec[0];
}