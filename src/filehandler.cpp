#include <iostream>
#include <fstream>
#include <iterator>
#include <cassert>
#include <algorithm>
#include <string>
#include <filesystem>
#include "filehandler.h"

using namespace std::string_literals;


std::vector<std::pair<Image, Eigen::Matrix<double, Eigen::Dynamic, 1>>> read_batch(ImageFile& images, LabelFile& labels, const int batch_size = 32) {
    // Read in a batch of {Image, Label} pairs from an ImageFile and a LabelFile.
    std::vector<std::pair<Image, Eigen::Matrix<double, Eigen::Dynamic, 1>>> batch_images;
    batch_images.reserve(images.num_entries);
    for (int i = 0; i < batch_size; ++i) {
        Eigen::Matrix<double, Eigen::Dynamic, 1> label_vec = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(10);
        label_vec(labels.read(i)) = 1.0;
        batch_images.emplace_back(images.read(i), label_vec);
    }
    return batch_images;
}
namespace {
    constexpr int swap_bytes(const size_t value) {
        // Swaps the bits in a 32-bit integer from big-endian to
        // little-endian or vice-versa. Thanks to user
        // Bordon on the MSDN forums.
        int retval = value & 0xFF;
        retval = (retval << 8) | ((value >> 8) & 0xFF);
        retval = (retval << 8) | ((value >> 16) & 0xFF);
        retval = (retval << 8) | ((value >> 24) & 0xFF);
        return retval;
    }
}

std::ostream& operator<< (std::ostream &stream, const Image& image) {
    // Makes a cute ASCII art version of the image and outputs
    // it to the ostream
    stream << "Outputting an image of size" << image.rows << "x" << image.cols << "\n";
    for (size_t y_coord = 0; y_coord < image.rows; ++y_coord){
        for (size_t x_coord = 0; x_coord < image.cols; ++x_coord) {
            auto pixel = image.read(x_coord, y_coord);
            if (0 <= pixel && pixel < 64) {
                stream <<  " ";
            } else if ( 64 <= pixel && pixel < 128) {
                stream <<  u8"\u2592";
            } else if ( 128 <= pixel && pixel < 196) {
                stream <<  u8"\u2593";
            } else if ( 196 <= pixel && pixel < 256) {
                stream << u8"\u2588";
            } else { 
                stream << u"X";
            }
        }
        stream << "\n";
    }
    return stream;
}
uint8_t Image::read(const size_t x_coord, const size_t y_coord) const{
    // Reads a value from the image at point (x, y)
    // and returns an integer representing the brightness
    // at that point.
    return data(x_coord, y_coord);
}

uint8_t Image::read(const size_t pixel) const{
    // Reads a value from the image of the nth pixel 
    // and returns an integer representing the brightness
    // at that point.
    return data(pixel);
}

Image::VecType Image::flatten() {
    Eigen::Map<Image::VecType> map(data.data(), data.size());
    return map;
}

Image::Image(Eigen::Ref<MatrType> input_data, size_t in_rows, size_t in_cols) : data{ input_data }, rows { in_rows }, cols{ in_cols }, total_pixels{ rows * cols } { }



ImageFile::ImageFile(const std::string& inname) {
    // Constructs the ImageFile object and does
    // some basic sanity checks such as "does the magic number match"
    // and "is the number of elements sane"
    // Also calculates the size of the header by reading the header
    // and gets the number of rows and columns from the header.
    filename = inname;
    std::cout << "Reading from " << filename << "\n";
    if (!std::filesystem::exists(filename)) {
        throw std::runtime_error("Could not find"s + inname);
    }
    infile = std::ifstream(filename, std::ios::in | std::ios::binary);
    if (!infile.is_open()) {
        throw std::runtime_error("Could not read file:"s + filename);
    }
    infile.seekg(0, std::ios::beg);
    infile.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = swap_bytes(magic_number);

    if (magic_number != 2051) {
        throw std::runtime_error("Magic number did not match. Expected 2051, got"s + std::to_string(magic_number));
    }
    infile.read(reinterpret_cast<char*>(&num_entries),4);
    num_entries = swap_bytes(num_entries);

    infile.read(reinterpret_cast<char*>(&num_rows), 4);
    num_rows = swap_bytes(num_rows);
    infile.read(reinterpret_cast<char*>(&num_cols), 4);
    num_cols = swap_bytes(num_cols);
    // Since we've just read the header, we can easily remember where
    // the offset is (it should be 8).
    header_offset = infile.tellg();
}

Image ImageFile::read(size_t position) {
    // Reads an image from the file.
    // Loops round the file if position is out of range.
    infile.clear();

    position = position % num_entries;
    std::streampos label_offset = position * num_rows * num_cols;
    infile.seekg(label_offset + header_offset, std::ios::beg);
    Image::MatrType matr = Image::MatrType::Zero(num_rows, num_cols);
    infile.read( (char*)matr.data(), num_rows * num_cols * sizeof(uint8_t));
    return { matr, num_rows, num_cols };
}

LabelFile::LabelFile(const std::string& inname) : filename{ inname } {
     // Constructs the ImageFile object and does
    // some basic sanity checks such as "does the magic number match"
    // and "is the number of elements sane"
    // Also calculates the size of the header by reading the header
    infile = std::ifstream(filename, std::ios::in | std::ios::binary);
    if (!infile.is_open()) {
        throw std::runtime_error("Could not read file:"s + filename);
    }

    infile.seekg(0, std::ios::beg);
    infile.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    magic_number = swap_bytes(magic_number);
    if (magic_number != 2049) {
        throw std::runtime_error("Magic number did not match. Expected 2051, got"s + std::to_string(magic_number));
    }
    infile.read(reinterpret_cast<char *>(&num_entries), sizeof(num_entries));
    num_entries = swap_bytes(num_entries);
    // Since we've just read the header, we can easily remember where
    // the offset is (it should be 8).
    header_offset = infile.tellg();
}

int LabelFile::read(uint32_t position){
    // Reads a number from 0-9 from the file,
    // at the position specified. Returns
    // a negative number if access outside
    // of the range.
    infile.clear();
    position = position % num_entries;
    std::streampos label_offset = position;
    infile.seekg(label_offset + header_offset, std::ios::beg);
    if (infile.bad()) {
        throw std::runtime_error("Bad label offset read."s);
    }
    uint8_t number;
    infile.read(reinterpret_cast<char *>(&number), sizeof(number));
    return static_cast<int>(number);
}
