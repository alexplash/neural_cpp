#include <filesystem>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <vector>

namespace {

std::string shell_quote(const std::string& value) {
    std::string quoted = "'";

    for (char ch : value) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted += ch;
        }
    }

    quoted += "'";
    return quoted;
}

int decode_exit_code(int status) {
    if (status == -1) {
        return -1;
    }

    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }

    return status;
}

void download_file(const std::string& gz_path, const std::string& file_name) {
    const std::vector<std::string> base_urls = {
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "https://yann.lecun.org/exdb/mnist/",
        "https://yann.lecun.com/exdb/mnist/"
    };

    std::ostringstream attempts;

    for (const auto& base_url : base_urls) {
        const std::string url = base_url + file_name;
        const std::string curl_cmd =
            "curl -fL --retry 3 --connect-timeout 10 -o " +
            shell_quote(gz_path) + " " + shell_quote(url);

        std::cout << "Downloading " << file_name << " from " << url << "..." << std::endl;

        const int curl_result = std::system(curl_cmd.c_str());
        if (curl_result == 0) {
            return;
        }

        std::filesystem::remove(gz_path);
        attempts << "\n- " << url << " (curl exit code " << decode_exit_code(curl_result) << ")";
    }

    throw std::runtime_error(
        "Failed to download " + file_name + ". Tried:" + attempts.str()
    );
}

} // namespace

struct MNISTSample {
    std::vector<float> image; // length 784, normalized 0.0 to 1.0
    int label;
};

int read_int(std::ifstream& file) {
    unsigned char bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);

    return (static_cast<int>(bytes[0]) << 24) |
           (static_cast<int>(bytes[1]) << 16) |
           (static_cast<int>(bytes[2]) << 8)  |
           (static_cast<int>(bytes[3]));
}

void download_mnist_if_needed(const std::string& data_dir) {
    std::filesystem::create_directories(data_dir);

    struct FileInfo {
        std::string gz_name;
        std::string raw_name;
    };

    std::vector<FileInfo> files = {
        {"train-images-idx3-ubyte.gz", "train-images-idx3-ubyte"},
        {"train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte"},
        {"t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte"},
        {"t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte"}
    };

    for (const auto& file : files) {
        std::string raw_path = data_dir + "/" + file.raw_name;
        std::string gz_path = data_dir + "/" + file.gz_name;

        if (std::filesystem::exists(raw_path)) {
            continue;
        }

        if (!std::filesystem::exists(gz_path)) {
            download_file(gz_path, file.gz_name);
        }

        std::string gunzip_cmd = "gunzip -f " + shell_quote(gz_path);

        std::cout << "Extracting " << file.gz_name << "..." << std::endl;

        int gunzip_result = std::system(gunzip_cmd.c_str());
        if (gunzip_result != 0) {
            throw std::runtime_error(
                "Failed to extract " + file.gz_name +
                " (gunzip exit code " + std::to_string(decode_exit_code(gunzip_result)) + ")"
            );
        }
    }
}

std::vector<MNISTSample> load_mnist(
    const std::string& images_path,
    const std::string& labels_path
) {
    std::ifstream images_file(images_path, std::ios::binary);
    std::ifstream labels_file(labels_path, std::ios::binary);

    if (!images_file.is_open()) {
        throw std::runtime_error("Could not open images file: " + images_path);
    }

    if (!labels_file.is_open()) {
        throw std::runtime_error("Could not open labels file: " + labels_path);
    }

    int images_magic = read_int(images_file);
    int num_images = read_int(images_file);
    int rows = read_int(images_file);
    int cols = read_int(images_file);

    int labels_magic = read_int(labels_file);
    int num_labels = read_int(labels_file);

    if (images_magic != 2051) {
        throw std::runtime_error("Invalid image file magic number.");
    }

    if (labels_magic != 2049) {
        throw std::runtime_error("Invalid label file magic number.");
    }

    if (num_images != num_labels) {
        throw std::runtime_error("Image count and label count do not match.");
    }

    std::vector<MNISTSample> dataset;
    dataset.reserve(num_images);

    for (int i = 0; i < num_images; i++) {
        std::vector<float> image;
        image.reserve(rows * cols);

        for (int j = 0; j < rows * cols; j++) {
            unsigned char pixel = 0;
            images_file.read(reinterpret_cast<char*>(&pixel), 1);
            image.push_back(static_cast<float>(pixel) / 255.0f);
        }

        unsigned char label = 0;
        labels_file.read(reinterpret_cast<char*>(&label), 1);

        dataset.push_back(MNISTSample{
            image,
            static_cast<int>(label)
        });
    }

    return dataset;
}
