#include <iostream>
#include "classes/tensor.cpp"
#include <memory>
#include <vector>

int main() {
    Tensor scalar_a(10.0);
    auto scalar_b = std::make_shared<Tensor>(5.0);
    std::shared_ptr<Tensor> scalar_scalar = scalar_a + scalar_b;
    std::cout << "scalar + scalar: " << *scalar_scalar << std::endl;

    Tensor scalar_c(10.0);
    auto vector_a = std::make_shared<Tensor>(std::vector<float>{1.0, 2.0, 3.0});
    std::shared_ptr<Tensor> scalar_vector = scalar_c + vector_a;
    std::cout << "scalar + 1d: " << *scalar_vector << std::endl;

    Tensor scalar_d(10.0);
    auto matrix_a = std::make_shared<Tensor>(
        std::vector<std::vector<float>>{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    std::shared_ptr<Tensor> scalar_matrix = scalar_d + matrix_a;
    std::cout << "scalar + 2d: " << *scalar_matrix << std::endl;

    Tensor vector_b(std::vector<float>{1.0, 2.0, 3.0});
    auto scalar_e = std::make_shared<Tensor>(10.0);
    std::shared_ptr<Tensor> vector_scalar = vector_b + scalar_e;
    std::cout << "1d + scalar: " << *vector_scalar << std::endl;

    Tensor matrix_b(std::vector<std::vector<float>>{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    auto scalar_f = std::make_shared<Tensor>(10.0);
    std::shared_ptr<Tensor> matrix_scalar = matrix_b + scalar_f;
    std::cout << "2d + scalar: " << *matrix_scalar << std::endl;

    Tensor vector_c(std::vector<float>{1.0, 2.0, 3.0});
    auto vector_d = std::make_shared<Tensor>(std::vector<float>{10.0, 20.0, 30.0});
    std::shared_ptr<Tensor> vector_vector = vector_c + vector_d;
    std::cout << "1d + 1d: " << *vector_vector << std::endl;

    Tensor matrix_c(std::vector<std::vector<float>>{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    auto matrix_d = std::make_shared<Tensor>(
        std::vector<std::vector<float>>{{10.0, 20.0, 30.0}, {40.0, 50.0, 60.0}});
    std::shared_ptr<Tensor> matrix_matrix = matrix_c + matrix_d;
    std::cout << "2d + 2d: " << *matrix_matrix << std::endl;

    return 0;
}
