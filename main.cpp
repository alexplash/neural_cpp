#include <iostream>
#include <array>
#include <filesystem>
#include <memory>
#include <vector>
#include <random>
#include <random>
#include <algorithm>
#include <type_traits>
#include "classes/tensor.cpp"
#include "data/mnist.cpp"
#include "utils/utils.cpp"

int main() {

    // float min_float = -1;
    // float max_float = 1;
    // float learning_rate = 0.0001;
    // float batch_size = 32;

    // auto input_x = std::make_shared<Tensor>(
    //     random_vector(784, min_float, max_float),
    //     false
    // );

    // auto weights_1 = std::make_shared<Tensor>(
    //     random_matrix(784, 512, min_float, max_float),
    //     true
    // );

    // auto bias_1 = std::make_shared<Tensor>(
    //     random_vector(512, min_float, max_float),
    //     true
    // );

    // auto weights_2 = std::make_shared<Tensor>(
    //     random_matrix(512, 512, min_float, max_float),
    //     true
    // );

    // auto bias_2 = std::make_shared<Tensor>(
    //     random_vector(512, min_float, max_float),
    //     true
    // );

    // auto weights_3 = std::make_shared<Tensor>(
    //     random_matrix(512, 10, min_float, max_float),
    //     true
    // );

    // auto bias_3 = std::make_shared<Tensor>(
    //     random_vector(10, min_float, max_float),
    //     true
    // );

    // std::vector<std::shared_ptr<Tensor>> params{
    //     weights_1,
    //     bias_1,
    //     weights_2,
    //     bias_2,
    //     weights_3,
    //     bias_3
    // };

    // std::shared_ptr<Tensor> h1 = *input_x * weights_1;
    // std::shared_ptr<Tensor> h1_b = *h1 + bias_1;
    // std::shared_ptr<Tensor> h1_relu = h1_b->relu();
    // std::shared_ptr<Tensor> h2 = *h1_relu * weights_2;
    // std::shared_ptr<Tensor> h2_b = *h2 + bias_2;
    // std::shared_ptr<Tensor> h2_relu = h2_b->relu();
    // std::shared_ptr<Tensor> h3 = *h2_relu * weights_3;
    // std::shared_ptr<Tensor> logits = *h3 + bias_3;

    // std::cout << "logits: " << *logits << std::endl;
    // std::cout << "forward pass completed successfully" << std::endl;

    // auto target = std::make_shared<Tensor>(
    //     std::vector<float>{0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    //     false
    // );

    // std::shared_ptr<Tensor> loss = *logits * target;

    // std::cout << "loss: " << *loss << std::endl;

    // loss->backward();
    // for (std::size_t i = 0; i < params.size(); i++) {
    //     params[i]->sgd_step(learning_rate, batch_size);
    //     params[i]->zero_grad();
    // }

    // return 0;

    std::string data_dir =
        (std::filesystem::path(NEURAL_CPP_SOURCE_DIR) / "data" / "mnist").string();

    download_mnist_if_needed(data_dir);

    std::vector<MNISTSample> train_data = load_mnist(
        data_dir + "/train-images-idx3-ubyte",
        data_dir + "/train-labels-idx1-ubyte"
    );

    std::vector<MNISTSample> test_data = load_mnist(
        data_dir + "/t10k-images-idx3-ubyte",
        data_dir + "/t10k-labels-idx1-ubyte"
    );

    return 0;
}
