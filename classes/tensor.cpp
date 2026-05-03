#include <cstddef>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <memory>
#include <cmath>

class Tensor : public std::enable_shared_from_this<Tensor> {
private:
    std::vector<float> _data;
    std::vector<std::size_t> _shape;
    std::vector<float> _grad;
    std::function<void(const std::vector<float>&)> _gradfn;
    std::vector<std::shared_ptr<Tensor>> _parents;
    bool _requires_grad;

    void _backward() {
        if (!_requires_grad) {
            return;
        }

        if (_gradfn) {
            _gradfn(_grad);
        }
        for (std::size_t i = 0; i < _parents.size(); i++) {
            _parents[i]->_backward();
        }

    }

public:
    Tensor(
        float data, 
        bool requires_grad = false, 
        std::function<void(const std::vector<float>&)> grad_fn = nullptr,
        std::vector<std::shared_ptr<Tensor>> parents = {}
    ) : _data{data},
        _shape{},
        _grad{},
        _gradfn{grad_fn},
        _parents{parents},
        _requires_grad{requires_grad}
    {
        if (_requires_grad) {
            zero_grad();
        }
    }

    Tensor(
        std::vector<float> data,
        bool requires_grad = false, 
        std::function<void(const std::vector<float>&)> grad_fn = nullptr,
        std::vector<std::shared_ptr<Tensor>> parents = {}
    ) : _data(data),
        _shape{data.size()},
        _grad{},
        _gradfn{grad_fn},
        _parents{parents},
        _requires_grad{requires_grad}
    {
        if (_requires_grad) {
            zero_grad();
        }
    }

    Tensor(
        std::vector<std::vector<float>> data,
        bool requires_grad = false, 
        std::function<void(const std::vector<float>&)> grad_fn = nullptr,
        std::vector<std::shared_ptr<Tensor>> parents = {}
    ) : _data{},
        _shape{},
        _grad{},
        _gradfn{grad_fn},
        _parents{parents},
        _requires_grad{requires_grad}
    {
        if (data.empty() || data[0].empty()) {
            throw std::invalid_argument("Tensor data cannot be empty.");
        }

        std::size_t rows = data.size();
        std::size_t cols = data[0].size();

        _shape = {rows, cols};

        for (std::size_t i = 0; i < rows; i++) {
            if (data[i].size() != cols) {
                throw std::invalid_argument("Dimensions are inconsistent.");
            }

            for (std::size_t j = 0; j < cols; j++) {
                _data.push_back(data[i][j]);
            }
        }

        if (_requires_grad) {
            zero_grad();
        }
    }

    float &item() {
        if (_data.size() != 1) {
            throw std::runtime_error("item() requires exactly one element.");
        }
        return _data[0];
    }

    float &operator()(std::size_t i) {
        if (_shape.size() != 1) {
            throw std::invalid_argument("Single indexing only works for 1D tensors.");
        }
        if (i >= _shape[0]) {
            throw std::out_of_range("Index out of bounds.");
        }
        return _data[i];
    }

    float &operator()(std::size_t i, std::size_t j) {
        if (_shape.size() != 2) {
            throw std::invalid_argument("Double indexing only works for 2D tensors.");
        }
        if (i >= _shape[0] || j >= _shape[1]) {
            throw std::out_of_range("Index out of bounds.");
        }
        return _data[i * _shape[1] + j];
    }

    std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> other) {
        // scalar + scalar
        if (_shape.size() == 0 && other -> shape().size() == 0) {
            float result = item() + other -> item();

            if (_requires_grad || other->requires_grad()) {
                std::shared_ptr<Tensor> self = shared_from_this();
                std::vector<std::shared_ptr<Tensor>> parents{self, other};
                std::function<void(const std::vector<float>&)> gradfn = 
                    [self, other](const std::vector<float>& grad_output) {
                        self->add_to_grad({grad_output[0]});
                        other->add_to_grad({grad_output[0]});
                    };
                return std::make_shared<Tensor>(result, true, gradfn, parents);
            }

            return std::make_shared<Tensor>(result);
        }

        // scalar + 1d
        if (_shape.size() == 0 && other -> shape().size() == 1) {
            std::vector<float> result;
            for (std::size_t i = 0; i < other->shape()[0]; i++) {
                result.push_back(item() + ((*other)(i)));
            }

            if (_requires_grad || other->requires_grad()) {
                std::shared_ptr<Tensor> self = shared_from_this();
                std::vector<std::shared_ptr<Tensor>> parents{self, other};
                std::function<void(const std::vector<float>&)> gradfn = 
                    [self, other](const std::vector<float>& grad_output) {
                        float grad_self = 0;
                        for (std::size_t i = 0; i < grad_output.size(); i++) {
                            grad_self += grad_output[i];
                        }
                        self->add_to_grad({grad_self});
                        other->add_to_grad(grad_output);
                    };
                return std::make_shared<Tensor>(result, true, gradfn, parents);
            }

            return std::make_shared<Tensor>(result);
        }

        // scalar + 2d
        if (_shape.size() == 0 && other -> shape().size() == 2) {
            std::vector<std::vector<float>> result;
            for (std::size_t i = 0; i < other->shape()[0]; i++) {
                std::vector<float> result_i;
                for (std::size_t j = 0; j < other->shape()[1]; j++) {
                    result_i.push_back(item() + ((*other)(i, j)));
                }
                result.push_back(result_i);
            }

            if (_requires_grad || other->requires_grad()) {
                std::shared_ptr<Tensor> self = shared_from_this();
                std::vector<std::shared_ptr<Tensor>> parents{self, other};
                std::function<void(const std::vector<float>&)> gradfn = 
                    [self, other](const std::vector<float>& grad_output) {
                        float grad_self = 0;
                        for (std::size_t i = 0; i < grad_output.size(); i++) {
                            grad_self += grad_output[i];
                        }
                        self->add_to_grad({grad_self});
                        other->add_to_grad(grad_output);
                    };
                return std::make_shared<Tensor>(result, true, gradfn, parents);
            }

            return std::make_shared<Tensor>(result);
        }

        // 1d + scalar
        if (_shape.size() == 1 && other -> shape().size() == 0) {
            std::vector<float> result;
            for (std::size_t i = 0; i < _shape[0]; i++) {
                result.push_back(operator()(i) + other->item());
            }

            if (_requires_grad || other->requires_grad()) {
                std::shared_ptr<Tensor> self = shared_from_this();
                std::vector<std::shared_ptr<Tensor>> parents{self, other};
                std::function<void(const std::vector<float>&)> gradfn = 
                    [self, other](const std::vector<float>& grad_output) {
                        float grad_other = 0;
                        for (std::size_t i = 0; i < grad_output.size(); i++) {
                            grad_other += grad_output[i];
                        }
                        self->add_to_grad(grad_output);
                        other->add_to_grad({grad_other});
                    };
                return std::make_shared<Tensor>(result, true, gradfn, parents);
            }

            return std::make_shared<Tensor>(result);
        }

        // 2d + scalar
        if (_shape.size() == 2 && other -> shape().size() == 0) {
            std::vector<std::vector<float>> result;
            for (std::size_t i = 0; i < _shape[0]; i++) {
                std::vector<float> result_i;
                for (std::size_t j = 0; j < _shape[1]; j++) {
                    result_i.push_back(operator()(i, j) + other->item());
                }
                result.push_back(result_i);
            }

            if (_requires_grad || other->requires_grad()) {
                std::shared_ptr<Tensor> self = shared_from_this();
                std::vector<std::shared_ptr<Tensor>> parents{self, other};
                std::function<void(const std::vector<float>&)> gradfn =
                    [self, other](const std::vector<float>& grad_output) {
                        float grad_other = 0;
                        for (std::size_t i = 0; i < grad_output.size(); i++) {
                            grad_other += grad_output[i];
                        }
                        self->add_to_grad(grad_output);
                        other->add_to_grad({grad_other});
                    };
                return std::make_shared<Tensor>(result, true, gradfn, parents);
            }

            return std::make_shared<Tensor>(result);
        }

        // 1d + 1d
        if (_shape[0] != other->shape()[0]) {
            throw std::invalid_argument("First dimensions are not equal.");
        }

        if (_shape.size() != other->shape().size()) {
            throw std::invalid_argument("Tensor dimensions must match for addition.");
        }

        
        std::vector<float> result;
        for (std::size_t i = 0; i < _shape[0]; i++) {
            result.push_back(operator()(i) + (*other)(i));
        }

        if (_requires_grad || other->requires_grad()) {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float>&)> gradfn =
                [self, other](const std::vector<float>& grad_output) {
                    self->add_to_grad(grad_output);
                    other->add_to_grad(grad_output);
                };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }

        return std::make_shared<Tensor>(result);
        
    }

    std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> other) {
        if (_shape.size() == 0 && other->shape().size() == 0) {
            throw std::invalid_argument("Both arguments must be atleast 1d for matmul.");
        }

        if (_shape[_shape.size() - 1] != other->shape()[0]) {
            throw std::invalid_argument(
                "Last dimension of first tensor doesn't equal first dimension of second tensor."
            );
        }

        // 1d * 1d = float
        if (_shape.size() == 1 && other->shape().size() == 1) {
            float result = 0;
            for (std::size_t i = 0; i < _shape[0]; i++) {
                result += (operator()(i) * (*other)(i));
            }

            if (_requires_grad || other->requires_grad()) {
                std::shared_ptr<Tensor> self = shared_from_this();
                std::vector<std::shared_ptr<Tensor>> parents{self, other};
                std::function<void(const std::vector<float>&)> gradfn =
                    [self, other](const std::vector<float>& grad_output) {
                        std::vector<float> grad_self;
                        std::vector<float> grad_other;
                        for (std::size_t i = 0; i < self->shape()[0]; i++) {
                            grad_self.push_back((*other)(i) * grad_output[0]);
                            grad_other.push_back((*self)(i) * grad_output[0]);
                        }
                        self->add_to_grad(grad_self);
                        other->add_to_grad(grad_other);
                    };
                return std::make_shared<Tensor>(result, true, gradfn, parents);
            }
            
            return std::make_shared<Tensor>(result);
        }

        // 2d * 1d = 1d
        if (_shape.size() == 2 && other->shape().size() == 1) {
            std::vector<float> result;
            for (std::size_t i = 0; i < _shape[0]; i++) {
                float result_i = 0;
                for (std::size_t j = 0; j < _shape[1]; j++) {
                    result_i += (operator()(i, j) * (*other)(j));
                }
                result.push_back(result_i);
            }

            if (_requires_grad || other->requires_grad()) {
                std::shared_ptr<Tensor> self = shared_from_this();
                std::vector<std::shared_ptr<Tensor>> parents{self, other};
                std::function<void(const std::vector<float>&)> gradfn =
                    [self, other](const std::vector<float>& grad_output) {
                        std::vector<float> grad_self;
                        
                        for (std::size_t i = 0; i < self->shape()[0]; i++) {
                            for (std::size_t j = 0; j < self->shape()[1]; j++) {
                                grad_self.push_back((*other)(j) * grad_output[i]);
                            }
                        }

                        std::vector<float> grad_other;
                        for (std::size_t i = 0; i < other->shape()[0]; i++) {
                            float grad_other_i = 0;
                            for (std::size_t j = 0; j < self->shape()[0]; j++) {
                                grad_other_i += ((*self)(j, i) * grad_output[j]);
                            }
                            grad_other.push_back(grad_other_i);
                        }

                        self->add_to_grad(grad_self);
                        other->add_to_grad(grad_other);
                    };
                return std::make_shared<Tensor>(result, true, gradfn, parents);
            }

            return std::make_shared<Tensor>(result);
        }

        // 1d * 2d = 1d
        if (_shape.size() == 1 && other->shape().size() == 2) {
            std::vector<float> result;
            for (std::size_t j = 0; j < other->shape()[1]; j++) {
                float result_j = 0;
                for (std::size_t i = 0; i < other->shape()[0]; i++) {
                    result_j += (operator()(i) * (*other)(i, j));
                }
                result.push_back(result_j);
            }

            if (_requires_grad || other->requires_grad()) {
                std::shared_ptr<Tensor> self = shared_from_this();
                std::vector<std::shared_ptr<Tensor>> parents{self, other};
                std::function<void(const std::vector<float> &)> gradfn =
                    [self, other](const std::vector<float> &grad_output) {
                        std::vector<float> grad_self;
                        for (std::size_t i = 0; i < self->shape()[0]; i++) {
                            float grad_self_i = 0;
                            for (std::size_t j = 0; j < other->shape()[1]; j++) {
                                grad_self_i += ((*other)(i, j) * grad_output[j]);
                            }
                            grad_self.push_back(grad_self_i);
                        }

                        std::vector<float> grad_other;
                        for (std::size_t i = 0; i < other->shape()[0]; i++) {
                            for (std::size_t j = 0; j < other->shape()[1]; j++) {
                                grad_other.push_back((*self)(i) * grad_output[j]);
                            }
                        }

                        self->add_to_grad(grad_self);
                        other->add_to_grad(grad_other);
                    };
                return std::make_shared<Tensor>(result, true, gradfn, parents);
            }

            return std::make_shared<Tensor>(result);
        }

        return std::make_shared<Tensor>(0);

    }

    std::shared_ptr<Tensor> relu() {

        // 1d
        std::vector<float> result;

        for (std::size_t i = 0; i < _shape[0]; i++) {
            float value = operator()(i);
            result.push_back(value > 0 ? value : 0);
        }

        if (_requires_grad) {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self};
            std::function<void(const std::vector<float> &)> gradfn =
                [self](const std::vector<float> &grad_output) {
                    std::vector<float> grad_self;

                    for (std::size_t i = 0; i < self->shape()[0]; i++) {
                        float curr_grad = (*self)(i) > 0 ? 1 : 0;
                        grad_self.push_back(grad_output[i] * curr_grad);
                    }

                    self->add_to_grad(grad_self);
                };
            
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }

        return std::make_shared<Tensor>(result);
    }

    std::shared_ptr<Tensor> log() {

        // 1d
        if (_shape.size() != 1) {
            throw std::runtime_error("log only supports scalar and 1D tensors.");
        }

        std::vector<float> result;

        for (std::size_t i = 0; i < _shape[0]; i++) {
            float value = operator()(i);
            if (value <= 0.0f) {
                throw std::runtime_error("log input must be positive");
            }
            result.push_back(std::log(value));
        }

        if (_requires_grad) {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self};
            std::function<void(const std::vector<float>&)> gradfn =
                [self](const std::vector<float>& grad_output) {
                    std::vector<float> grad_self;

                    for (std::size_t i = 0; i < self->shape()[0]; i++) {
                        grad_self.push_back(grad_output[i] * (1.0f / (*self)(i)));
                    }

                    self->add_to_grad(grad_self);
                };

            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }

        return std::make_shared<Tensor>(result);
        
    }

    std::shared_ptr<Tensor> softmax() {
        if (_shape.size() != 1) {
            throw std::runtime_error("softmax requires 1d tensor");
        }

        float max_value = operator()(0);
        for (std::size_t i = 0; i < _shape[0]; i++) {
            if (operator()(i) > max_value) {
                max_value = operator()(i);
            }
        }

        std::vector<float> exp_values;
        float sum_exp = 0;
        for (std::size_t i = 0; i < _shape[0]; i++) {
            float exp_value = std::exp(operator()(i) - max_value);
            exp_values.push_back(exp_value);
            sum_exp += exp_value;
        }

        std::vector<float> result;
        for (std::size_t i = 0; i < _shape[0]; i++) {
            result.push_back(exp_values[i] / sum_exp);
        }

        if (_requires_grad) {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self};
            std::function<void(const std::vector<float>&)> gradfn =
                [self, result](const std::vector<float>& grad_output) {
                    std::vector<float> grad_self;

                    float dot = 0;
                    for (std::size_t j = 0; j < result.size(); j++) {
                        dot += grad_output[j] * result[j];
                    }

                    for (std::size_t i = 0; i < result.size(); i++) {
                        grad_self.push_back(result[i] * (grad_output[i] - dot));
                    }

                    self->add_to_grad(grad_self);
                };
            
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }

        return std::make_shared<Tensor>(result);
    }

    std::shared_ptr<Tensor> neg() {
        // scalar
        if (_shape.size() == 0) {
            float result = -1 * item();

            if (_requires_grad) {
                std::shared_ptr<Tensor> self = shared_from_this();
                std::vector<std::shared_ptr<Tensor>> parents{self};
                std::function<void(const std::vector<float>&)> gradfn =
                    [self](const std::vector<float>& grad_output) {
                        self->add_to_grad({-1 * grad_output[0]});
                    };
                
                return std::make_shared<Tensor>(result, true, gradfn, parents);
            }

            return std::make_shared<Tensor>(result);
        }

        // 1d
        if (_shape.size() != 1) {
            throw std::runtime_error("negative transformation only supports scalar and 1d.");
        }

        std::vector<float> result;
        for (std::size_t i = 0; i < _shape[0]; i++) {
            result.push_back(-1 * operator()(i));
        }

        if (_requires_grad) {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self};
            std::function<void(const std::vector<float>&)> gradfn =
                [self](const std::vector<float>& grad_output) {
                    std::vector<float> grad_self;

                    for (std::size_t i = 0; i < self->shape()[0]; i++) {
                        grad_self.push_back(-1 * grad_output[i]);
                    }

                    self->add_to_grad(grad_self);
                };

            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }

        return std::make_shared<Tensor>(result);
    }

    const std::vector<std::size_t> &shape() {
        return _shape;
    }

    const bool &requires_grad() {
        return _requires_grad;
    }

    const std::vector<float> &grad() {
        return _grad;
    }

    void add_to_grad(const std::vector<float>& grad_update) {
        if (!_requires_grad) {
            return;
        }
        if (_grad.size() != grad_update.size()) {
            throw std::runtime_error("Gradient shape mismatch during accumulation");
        }

        for (std::size_t i = 0; i < _grad.size(); i++) {
            _grad[i] += grad_update[i];
        }
    }

    void zero_grad() {
        _grad = std::vector<float>(_data.size(), 0);
    }

    void backward() {
        if (!_requires_grad) {
            throw std::runtime_error("Element does not require a grad.");
        }
        if (_shape.size() != 0) {
            throw std::runtime_error("Grad can only be calculated for scalar outputs");
        }

        _grad = {1.0};
        _backward();
    }

    void sgd_step(float& learning_rate, float& batch_size) {
        if (!_requires_grad) {
            return;
        }
        if (_data.size() != _grad.size()) {
            throw std::runtime_error("Data/gradient size mismatch during SGD step.");
        }

        for (std::size_t i = 0; i < _data.size(); i++) {
            _data[i] -= learning_rate * (_grad[i] / batch_size);
        }
    }

};
