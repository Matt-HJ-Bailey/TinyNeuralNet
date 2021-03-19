#include <algorithm>
#include <array>
#include <filesystem>
#include <random>
#include <utility>

#include "filehandler.h"
#include "activation_functions.h"
#include "multilayerperceptron.h"

#ifdef _WIN32
    #include <Windows.h>
#endif

int main(int argc, char** argv) {
    // Allow Windows to output Unicode symbols.
    // We need this to display numbers in the terminal.
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    setvbuf(stdout, nullptr, _IOFBF, 1000);
#endif

    std::array<std::mt19937::result_type, std::mt19937::state_size> seed_arr;
    std::random_device rd;
    std::generate(std::begin(seed_arr), std::end(seed_arr), std::ref(rd));
    std::seed_seq seeds(std::begin(seed_arr), std::end(seed_arr));
    std::mt19937 generator(seeds);

    LabelFile labels("./train-labels.idx1-ubyte"); 
    ImageFile images("./train-images.idx3-ubyte"); 

    LabelFile val_labels("./t10k-labels.idx1-ubyte");
    ImageFile val_images("./t10k-images.idx3-ubyte");
    
    auto val_batch = read_batch(val_images, val_labels, 100);

    // This is where the model is set up.
    MultiLayerPerceptron<Sigmoid> MLP(generator, { 28 * 28, 16, 16, 10 });


    for (int iter = 0; iter < 10000; ++iter) {
        auto batch_images = read_batch(images, labels, 64);

        auto [loss, accuracy, val_loss, val_accuracy] = MLP.fit(batch_images, val_batch, 0.01);
        std::cout << "At iteration " << iter << ", loss = " << loss << ", accuracy = " << accuracy << ", val loss = " << val_loss << ", val_accuracy = " << val_accuracy << "\n";
    }
    
}
