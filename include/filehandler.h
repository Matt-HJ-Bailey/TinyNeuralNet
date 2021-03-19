#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>

class Image {
    public:
        using MatrType = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>;
        using VecType = Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>;
        const size_t cols{28};
        const size_t rows{28};
        const size_t total_pixels{rows * cols};
        uint8_t read(size_t, size_t) const;
        uint8_t read(size_t) const;
        VecType flatten();
        Image(Eigen::Ref<MatrType>, size_t, size_t);
        friend std::ostream& operator<< (std::ostream &stream, const Image& image);
    private:
        MatrType data{ MatrType::Zero(rows, cols) };

};

class ImageFile {
    private:
        std::ifstream infile;
        // This is the size of the magic number and the number of entries;
        std::streampos header_offset;
        uint32_t magic_number;
        uint32_t num_rows;
        uint32_t num_cols;
    public:
        size_t num_entries;
        Image read(size_t position);
        std::string filename;
        ImageFile(const std::string&);
};

class LabelFile {
    private:
        std::ifstream infile;
        // This is the size of the magic number and number of entries
        std::streampos header_offset;
        uint32_t magic_number;
        uint32_t num_entries;
    public:
        int read(uint32_t position);
        std::string filename;
        LabelFile(const std::string&);
};

std::vector<std::pair<Image, Eigen::Matrix<double, Eigen::Dynamic, 1>>> read_batch(ImageFile& images, LabelFile& labels, const int batch_size);