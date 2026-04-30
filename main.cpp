#include <iostream>
#include <memory>
#include <vector>
#include "classes/tensor.cpp"

void print_grad(const std::string& name, std::shared_ptr<Tensor> t) {
    std::cout << name << ".grad = [";
    const std::vector<float>& grad = t->grad();

    for (std::size_t i = 0; i < grad.size(); i++) {
        std::cout << grad[i];
        if (i + 1 < grad.size()) {
            std::cout << ", ";
        }
    }

    std::cout << "]" << std::endl;
}

int main() {
    /*
        Build a small computational graph:

            a · b = dot_1          // 1d * 1d -> scalar
            M * v = mv             // 2d * 1d -> 1d
            mv · w = dot_2         // 1d * 1d -> scalar

            loss = dot_1 + dot_2 + bias

        Then call:

            loss->backward();

        This should backpropagate gradients through the graph.
    */

    auto a = std::make_shared<Tensor>(
        std::vector<float>{1.0, 2.0, 3.0},
        true
    );

    auto b = std::make_shared<Tensor>(
        std::vector<float>{4.0, 5.0, 6.0},
        true
    );

    auto M = std::make_shared<Tensor>(
        std::vector<std::vector<float>>{
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0}
        },
        true
    );

    auto v = std::make_shared<Tensor>(
        std::vector<float>{7.0, 8.0, 9.0},
        true
    );

    auto w = std::make_shared<Tensor>(
        std::vector<float>{10.0, 20.0},
        true
    );

    auto bias = std::make_shared<Tensor>(
        2.0,
        true
    );

    std::shared_ptr<Tensor> dot_1 = *a * b;
    std::cout << "dot_1 = a * b: " << *dot_1 << std::endl;

    std::shared_ptr<Tensor> mv = *M * v;
    std::cout << "mv = M * v: " << *mv << std::endl;

    std::shared_ptr<Tensor> dot_2 = *mv * w;
    std::cout << "dot_2 = mv * w: " << *dot_2 << std::endl;

    std::shared_ptr<Tensor> partial_loss = *dot_1 + dot_2;
    std::cout << "partial_loss = dot_1 + dot_2: " << *partial_loss << std::endl;

    std::shared_ptr<Tensor> loss = *partial_loss + bias;
    std::cout << "loss = partial_loss + bias: " << *loss << std::endl;

    loss->backward();

    std::cout << "\nGradients:" << std::endl;
    print_grad("a", a);
    print_grad("b", b);
    print_grad("M", M);
    print_grad("v", v);
    print_grad("w", w);
    print_grad("bias", bias);

    return 0;
}