#include <iostream>

#include "DigitScanner.hpp"

typedef std::chrono::time_point<std::chrono::high_resolution_clock> chrono_clock;

void call_from_main();
void print_elapsed_time(chrono_clock);

int main(int argc, char **argv) {

    /* enable all numerical exceptions, for debugging */
    // call_from_main();

    chrono_clock begin = std::chrono::high_resolution_clock::now();
    srand(static_cast<unsigned int>(time(NULL)));
    
    /* create the DigitScanner */
    int nodes[] = {784, 200, 10};
    DigitScanner<float> dgs(nodes, 2, false);
    
    /* files and folders */
    std::string path_data       = "/Users/deiss/Documents/Programmation/C++/DigitScanner/mnist_data/";
    std::string path_folder_ANN = "/Users/deiss/Documents/Programmation/C++/DigitScanner/";
    std::string filename_ANN    = "dgs_params_800.txt";
    
    /* actions */
    // dgs.load(path_folder_ANN, path_ANN);
    dgs.train(path_data, 60000, 0, 1, 10, 0.5, 0);
    dgs.test(path_data, 10000, 0);
    // dgs.save(path_folder_ANN, path_ANN);
    
    print_elapsed_time(begin);
 
    return 0;
    
}

/* computes and print execution time */
void print_elapsed_time(chrono_clock begin) {
    chrono_clock end = std::chrono::high_resolution_clock::now();
    auto dur = end - begin;
    auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
    std::cout << static_cast<double>(ms)/1000 << " s" << std::endl;
}
