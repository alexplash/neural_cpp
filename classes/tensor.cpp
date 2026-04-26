#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

class Tensor {
private:
    std::vector<float> _data;
    std::vector<std::size_t> _shape;
    std::vector<std::size_t> _stride;

public:
    Tensor(float data) : _data{data}, _shape{}, _stride{} {}

    Tensor(std::vector<float> data) : _data(data), _shape{data.size()}, _stride{1} {}

    Tensor(std::vector<std::vector<float>> data) {
        if (data.empty() || data[0].empty()) {
            throw std::invalid_argument("Tensor data cannot be empty.");
        }

        std::size_t rows = data.size();
        std::size_t cols = data[0].size();

        _shape = {rows, cols};
        _stride = {cols, 1};

        for (std::size_t i = 0; i < rows; i++) {
            if (data[i].size() != cols) {
                throw std::invalid_argument("Dimensions are inconsistent.");
            }

            for (std::size_t j = 0; j < cols; j++) {
                _data.push_back(data[i][j]);
            }
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
        return _data[i * _stride[0] + j * _stride[1]];
    }

    std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> other) {
        // scalar + scalar
        if (_shape.size() == 0 && other -> shape().size() == 0) {
            float result = item() + other -> item();
            return std::make_shared<Tensor>(result);
        }

        // scalar + 1d
        if (_shape.size() == 0 && other -> shape().size() == 1) {
            std::vector<float> result;
            for (std::size_t i = 0; i < other->shape()[0]; i++) {
                result.push_back(item() + ((*other)(i)));
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
            return std::make_shared<Tensor>(result);
        }

        // 1d + scalar
        if (_shape.size() == 1 && other -> shape().size() == 0) {
            std::vector<float> result;
            for (std::size_t i = 0; i < _shape[0]; i++) {
                result.push_back(operator()(i) + other->item());
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
            return std::make_shared<Tensor>(result);
        }

        // 1d + 1d
        if (_shape[0] != other->shape()[0]) {
            throw std::invalid_argument("First dimensions are not equal.");
        }

        if (_shape.size() != other->shape().size()) {
            throw std::invalid_argument("Tensor dimensions must match for addition.");
        }

        if (_shape.size() == 1) {
            std::vector<float> result;
            for (std::size_t i = 0; i < _shape[0]; i++) {
                result.push_back(operator()(i) + (*other)(i));
            }
            return std::make_shared<Tensor>(result);
        }
        // 2d + 2d
        else {
            if (_shape[1] != other->shape()[1]) {
                throw std::invalid_argument("Second dimensions are not equal.");
            }
            std::vector<std::vector<float>> result;
            for (std::size_t i = 0; i < _shape[0]; i++) {
                std::vector<float> result_i;
                for (std::size_t j = 0; j < _shape[1]; j++) {
                    result_i.push_back(operator()(i, j) + (*other)(i, j));
                }
                result.push_back(result_i);
            }
            return std::make_shared<Tensor>(result);
        }
        
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
            return std::make_shared<Tensor>(result);
        }

        // 2d * 2d = 2d
        std::vector<std::vector<float>> result;
        for (std::size_t i = 0; i < _shape[0]; i++) {
            std::vector<float> row;
            for (std::size_t j = 0; j < other->shape()[1]; j++) {
                float result_ij = 0;
                for (std::size_t k = 0; k < _shape[1]; k++) {
                    result_ij += (operator()(i, k) * (*other)(k, j));
                }
                row.push_back(result_ij);
            }
            result.push_back(row);
        }
        return std::make_shared<Tensor>(result); 

    }

    const std::vector<std::size_t> &shape() {
        return _shape;
    }

    const std::vector<std::size_t> &stride() {
        return _stride;
    }

    friend std::ostream &operator<<(std::ostream &os, Tensor &obj) {
        if (obj._shape.empty()) {
            os << obj._data[0];
            return os;
        }

        if (obj._shape.size() == 1) {
            os << "[";
            for (std::size_t i = 0; i < obj._shape[0]; i++) {
                os << obj._data[i];
                if (i + 1 < obj._shape[0]) {
                    os << ", ";
                }
            }
            os << "]";
            return os;
        }

        os << "[";
        for (std::size_t i = 0; i < obj._shape[0]; i++) {
            os << "[";
            for (std::size_t j = 0; j < obj._shape[1]; j++) {
                os << obj(i, j);
                if (j + 1 < obj._shape[1]) {
                    os << ", ";
                }
            }
            os << "]";
            if (i + 1 < obj._shape[0]) {
                os << ", ";
            }
        }
        os << "]";
        return os;
    }
};
