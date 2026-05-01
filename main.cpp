#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include "classes/tensor.cpp"

std::vector<float> random_vector(std::size_t size, float min = -1.0f, float max = 1.0f) {
    static std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(min, max);

    std::vector<float> values;
    values.reserve(size);

    for (std::size_t i = 0; i < size; i++) {
        values.push_back(dist(gen));
    }

    return values;
}

std::vector<std::vector<float>> random_matrix(
    std::size_t rows,
    std::size_t cols,
    float min = -1.0f,
    float max = 1.0f
) {
    static std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(min, max);

    std::vector<std::vector<float>> values;
    values.reserve(rows);

    for (std::size_t i = 0; i < rows; i++) {
        std::vector<float> row;
        row.reserve(cols);

        for (std::size_t j = 0; j < cols; j++) {
            row.push_back(dist(gen));
        }

        values.push_back(row);
    }

    return values;
}

void print_grad_preview(const std::string& name, std::shared_ptr<Tensor> t, std::size_t max_items = 10) {
    const std::vector<float>& grad = t->grad();

    std::cout << name << ".grad shape numel = " << grad.size() << std::endl;
    std::cout << name << ".grad preview = [";

    std::size_t limit = std::min(max_items, grad.size());

    for (std::size_t i = 0; i < limit; i++) {
        std::cout << grad[i];
        if (i + 1 < limit) {
            std::cout << ", ";
        }
    }

    if (grad.size() > max_items) {
        std::cout << ", ...";
    }

    std::cout << "]" << std::endl;
}

int main() {
    /*
    input_x:   (784)

    weights_1: (784, 512)
    bias_1:    (512)
    ReLU

    weights_2: (512, 512)
    bias_2:    (512)
    ReLU

    weights_3: (512, 10)
    bias_3:    (10)

    logits = (ReLU((ReLU((input_x * weights_1) + bias_1) * weights_2) + bias_2) * weights_3) + bias_3
    */

    auto input_x = std::make_shared<Tensor>(
        random_vector(784),
        false
    );

    auto weights_1 = std::make_shared<Tensor>(
        random_matrix(784, 512),
        true
    );

    auto bias_1 = std::make_shared<Tensor>(
        random_vector(512),
        true
    );

    auto weights_2 = std::make_shared<Tensor>(
        random_matrix(512, 512),
        true
    );

    auto bias_2 = std::make_shared<Tensor>(
        random_vector(512),
        true
    );

    auto weights_3 = std::make_shared<Tensor>(
        random_matrix(512, 10),
        true
    );

    auto bias_3 = std::make_shared<Tensor>(
        random_vector(10),
        true
    );

    std::shared_ptr<Tensor> h1 = *input_x * weights_1;
    h1 = *h1 + bias_1;
    h1 = h1->relu();

    std::shared_ptr<Tensor> h2 = *h1 * weights_2;
    h2 = *h2 + bias_2;
    h2 = h2->relu();

    std::shared_ptr<Tensor> logits = *h2 * weights_3;
    logits = *logits + bias_3;

    std::cout << "logits: " << *logits << std::endl;
    std::cout << "forward pass completed successfully" << std::endl;

    /*
    Create a fake scalar loss.

    logits: (10)
    target: (10)

    loss = logits * target

    Since 1D * 1D returns a scalar dot product, loss is scalar,
    so loss->backward() is valid.
    */

    auto target = std::make_shared<Tensor>(
        std::vector<float>{0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        false
    );

    std::shared_ptr<Tensor> loss = *logits * target;

    std::cout << "loss: " << *loss << std::endl;

    loss->backward();

    std::cout << "\nbackward pass completed successfully" << std::endl;

    std::cout << "\nGradient previews:" << std::endl;
    print_grad_preview("weights_1", weights_1);
    print_grad_preview("bias_1", bias_1);
    print_grad_preview("weights_2", weights_2);
    print_grad_preview("bias_2", bias_2);
    print_grad_preview("weights_3", weights_3);
    print_grad_preview("bias_3", bias_3);

    return 0;
}