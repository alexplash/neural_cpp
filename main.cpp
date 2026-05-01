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

    // load in MNIST data
    // ----------------------------------------------------------------------------------------------
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
    // ----------------------------------------------------------------------------------------------

    // define hyper parameters
    // ----------------------------------------------------------------------------------------------
    float min_float = -0.05f;
    float max_float = 0.05f;
    float learning_rate = 0.0005;
    float batch_size = 32;
    std::size_t epochs = 10;
    // ----------------------------------------------------------------------------------------------

    // define neural network & params
    // ----------------------------------------------------------------------------------------------
    std::shared_ptr<Tensor> weights_1 = std::make_shared<Tensor>(
        random_matrix(784, 512, min_float, max_float),
        true
    );
    std::shared_ptr<Tensor> bias_1 = std::make_shared<Tensor>(
        random_vector(512, 0.0f, 0.0f),
        true
    );
    std::shared_ptr<Tensor> weights_2 = std::make_shared<Tensor>(
        random_matrix(512, 512, min_float, max_float),
        true
    );
    std::shared_ptr<Tensor> bias_2 = std::make_shared<Tensor>(
        random_vector(512, 0.0f, 0.0f),
        true
    );
    std::shared_ptr<Tensor> weights_3 = std::make_shared<Tensor>(
        random_matrix(512, 10, min_float, max_float),
        true
    );
    std::shared_ptr<Tensor> bias_3 = std::make_shared<Tensor>(
        random_vector(10, 0.0f, 0.0f),
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
    // ----------------------------------------------------------------------------------------------

    
    // training loop
    // ----------------------------------------------------------------------------------------------
    std::cout << "Starting training..." << std::endl;

    for (std::size_t epoch = 0; epoch < epochs; epoch++) {
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs << " started" << std::endl;

        float curr_batch_count = 0;
        float epoch_loss = 0.0f;
        std::size_t step_count = 0;

        for (std::size_t train_index = 0; train_index < train_data.size(); train_index++) {
            curr_batch_count++;
            std::shared_ptr<Tensor> input_x = std::make_shared<Tensor>(
                train_data[train_index].image,
                false
            );

            // forward pass
            std::shared_ptr<Tensor> h1 = *input_x * weights_1;
            std::shared_ptr<Tensor> h1_b = *h1 + bias_1;
            std::shared_ptr<Tensor> h1_relu = h1_b->relu();
            std::shared_ptr<Tensor> h2 = *h1_relu * weights_2;
            std::shared_ptr<Tensor> h2_b = *h2 + bias_2;
            std::shared_ptr<Tensor> h2_relu = h2_b->relu();
            std::shared_ptr<Tensor> h3 = *h2_relu * weights_3;
            std::shared_ptr<Tensor> logits = *h3 + bias_3;

            // cross entrupy loss
            std::vector<float> target_vector(10, 0.0f);
            target_vector[train_data[train_index].label] = 1.0f;
            std::shared_ptr<Tensor> target = std::make_shared<Tensor>(target_vector, false);
            std::shared_ptr<Tensor> probs = logits->softmax();
            std::shared_ptr<Tensor> log_probs = probs->log();
            std::shared_ptr<Tensor> dot_log_probs = *log_probs * target;
            std::shared_ptr<Tensor> loss = dot_log_probs->neg();

            epoch_loss += loss->item();

            // backpropagation & optimization
            loss->backward();

            if (curr_batch_count >= batch_size) {
                for (std::size_t i = 0; i < params.size(); i++) {
                    params[i]->sgd_step(learning_rate, curr_batch_count);
                    params[i]->zero_grad();
                }

                step_count++;
                if (step_count % 50 == 0) {
                    std::cout << "  step " << step_count
                              << " / sample " << train_index + 1
                              << " / avg loss so far: " << (epoch_loss / (train_index + 1))
                              << std::endl;
                }

                curr_batch_count = 0;
            }
        }

        if (curr_batch_count > 0) {
            for (std::size_t i = 0; i < params.size(); i++) {
                params[i]->sgd_step(learning_rate, curr_batch_count);
                params[i]->zero_grad();
            }

            step_count++;
                if (step_count % 50 == 0) {
                    std::cout << "  step " << step_count
                              << " / sample " << train_data.size()
                              << " / avg loss so far: " << (epoch_loss / (train_data.size()))
                              << std::endl;
            }
        }

        std::cout << "Epoch " << (epoch + 1)
            << " finished. avg loss: " << (epoch_loss / train_data.size())
            << std::endl;

    }

    std::cout << "Training complete." << std::endl;
    // ----------------------------------------------------------------------------------------------


    // validation loop
    // ----------------------------------------------------------------------------------------------
    std::cout << "Starting validation..." << std::endl;

    float total_images = static_cast<float>(test_data.size());
    float total_correct = 0.0f;
    for (std::size_t test_index = 0; test_index < test_data.size(); test_index++) {
        std::shared_ptr<Tensor> input_x = std::make_shared<Tensor>(
            test_data[test_index].image,
            false
        );

        int target_index = test_data[test_index].label;

        // forward pass
        std::shared_ptr<Tensor> h1 = *input_x * weights_1;
        std::shared_ptr<Tensor> h1_b = *h1 + bias_1;
        std::shared_ptr<Tensor> h1_relu = h1_b->relu();
        std::shared_ptr<Tensor> h2 = *h1_relu * weights_2;
        std::shared_ptr<Tensor> h2_b = *h2 + bias_2;
        std::shared_ptr<Tensor> h2_relu = h2_b->relu();
        std::shared_ptr<Tensor> h3 = *h2_relu * weights_3;
        std::shared_ptr<Tensor> logits = *h3 + bias_3;

        std::size_t argmax = 0;
        for (std::size_t i = 0; i < logits->shape()[0]; i++) {
            if ((*logits)(i) > (*logits)(argmax)) {
                argmax = i;
            }
        }

        if (argmax == static_cast<std::size_t>(target_index)) {
            total_correct++;
        }
    }

    float accuracy = total_correct / total_images;

    std::cout << "Completed validation / " << "accuracy = " << (accuracy * 100) << "%" << std::endl;
    // ----------------------------------------------------------------------------------------------


    return 0;
}
