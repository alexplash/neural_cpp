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

    float min_float = -1;
    float max_float = 1;
    float learning_rate = 0.0001;
    float batch_size = 32;
    std::size_t epochs = 20;

    std::shared_ptr<Tensor> weights_1 = std::make_shared<Tensor>(
        random_matrix(784, 512, min_float, max_float),
        true
    );
    std::shared_ptr<Tensor> bias_1 = std::make_shared<Tensor>(
        random_vector(512, min_float, max_float),
        true
    );
    std::shared_ptr<Tensor> weights_2 = std::make_shared<Tensor>(
        random_matrix(512, 512, min_float, max_float),
        true
    );
    std::shared_ptr<Tensor> bias_2 = std::make_shared<Tensor>(
        random_vector(512, min_float, max_float),
        true
    );
    std::shared_ptr<Tensor> weights_3 = std::make_shared<Tensor>(
        random_matrix(512, 10, min_float, max_float),
        true
    );
    std::shared_ptr<Tensor> bias_3 = std::make_shared<Tensor>(
        random_vector(10, min_float, max_float),
        true
    );

    std::vector<std::shared_ptr<Tensor>> params{
        weights_1,
        bias_1,
        weights_2,
        bias_2,
        weights_3,
        bias_3
    };

    for (std::size_t epoch = 0; epoch < epochs; epoch++) {
        float curr_batch_count = 0;

        for (std::size_t train_index = 0; train_index < train_data.size(); train_index++) {
            curr_batch_count++;
            std::shared_ptr<Tensor> input_x = std::make_shared<Tensor>(
                train_data[train_index].image,
                false
            );

            std::shared_ptr<Tensor> h1 = *input_x * weights_1;
            std::shared_ptr<Tensor> h1_b = *h1 + bias_1;
            std::shared_ptr<Tensor> h1_relu = h1_b->relu();
            std::shared_ptr<Tensor> h2 = *h1_relu * weights_2;
            std::shared_ptr<Tensor> h2_b = *h2 + bias_2;
            std::shared_ptr<Tensor> h2_relu = h2_b->relu();
            std::shared_ptr<Tensor> h3 = *h2_relu * weights_3;
            std::shared_ptr<Tensor> logits = *h3 + bias_3;

            std::vector<float> target_vector(10, 0.0f);
            target_vector[train_data[train_index].label] = 1.0f;
            std::shared_ptr<Tensor> target = std::make_shared<Tensor>(target_vector, false);
            std::shared_ptr<Tensor> probs = logits->softmax();
            std::shared_ptr<Tensor> log_probs = probs->log();
            std::shared_ptr<Tensor> dot_log_probs = *log_probs * target;
            std::shared_ptr<Tensor> loss = dot_log_probs->neg();

            loss->backward();

            if (curr_batch_count >= batch_size) {
                for (std::size_t i = 0; i < params.size(); i++) {
                    params[i]->sgd_step(learning_rate, curr_batch_count);
                    params[i]->zero_grad();
                }
                curr_batch_count = 0;
            }
        }

        if (curr_batch_count > 0) {
            for (std::size_t i = 0; i < params.size(); i++) {
                params[i]->sgd_step(learning_rate, curr_batch_count);
                params[i]->zero_grad();
            }
        }
    }


    return 0;
}
