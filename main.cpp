#include <iostream>
#include <memory>
#include <vector>
#include "classes/tensor.cpp"

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

    Tensor matmul_vector_a(std::vector<float>{1.0, 2.0, 3.0});
    auto matmul_vector_b = std::make_shared<Tensor>(std::vector<float>{4.0, 5.0, 6.0});
    std::shared_ptr<Tensor> vector_vector_product = matmul_vector_a * matmul_vector_b;
    std::cout << "1d * 1d: " << *vector_vector_product << std::endl;

    Tensor matmul_matrix_a(std::vector<std::vector<float>>{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    auto matmul_vector_c = std::make_shared<Tensor>(std::vector<float>{7.0, 8.0, 9.0});
    std::shared_ptr<Tensor> matrix_vector_product = matmul_matrix_a * matmul_vector_c;
    std::cout << "2d * 1d: " << *matrix_vector_product << std::endl;

    Tensor matmul_vector_d(std::vector<float>{1.0, 2.0});
    auto matmul_matrix_b = std::make_shared<Tensor>(
        std::vector<std::vector<float>>{{3.0, 4.0, 5.0}, {6.0, 7.0, 8.0}});
    std::shared_ptr<Tensor> vector_matrix_product = matmul_vector_d * matmul_matrix_b;
    std::cout << "1d * 2d: " << *vector_matrix_product << std::endl;

    Tensor matmul_matrix_c(std::vector<std::vector<float>>{{1.0, 2.0}, {3.0, 4.0}});
    auto matmul_matrix_d = std::make_shared<Tensor>(
        std::vector<std::vector<float>>{{5.0, 6.0}, {7.0, 8.0}});
    std::shared_ptr<Tensor> matrix_matrix_product = matmul_matrix_c * matmul_matrix_d;
    std::cout << "2d * 2d: " << *matrix_matrix_product << std::endl;

    try {
        Tensor scalar_g(2.0);
        auto scalar_h = std::make_shared<Tensor>(3.0);
        std::shared_ptr<Tensor> scalar_scalar_product = scalar_g * scalar_h;
        std::cout << "scalar * scalar: " << *scalar_scalar_product << std::endl;
    } catch (const std::invalid_argument &e) {
        std::cout << "scalar * scalar error: " << e.what() << std::endl;
    }

    try {
        Tensor invalid_matrix(std::vector<std::vector<float>>{{1.0, 2.0, 3.0}});
        auto invalid_vector = std::make_shared<Tensor>(std::vector<float>{4.0, 5.0});
        std::shared_ptr<Tensor> invalid_product = invalid_matrix * invalid_vector;
        std::cout << "invalid 2d * 1d: " << *invalid_product << std::endl;
    } catch (const std::invalid_argument &e) {
        std::cout << "invalid 2d * 1d error: " << e.what() << std::endl;
    }

    return 0;
}
