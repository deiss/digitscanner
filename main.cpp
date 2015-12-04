#include <iostream>

#include "DigitScanner.hpp"
#include "Window.hpp"

typedef std::chrono::time_point<std::chrono::high_resolution_clock> chrono_clock;

void call_from_main();
void print_elapsed_time(chrono_clock);

int main(int argc, char **argv) {
    /* enable all numerical exceptions, for debugging */
    // call_from_main();

    /* initializations */
    chrono_clock begin = std::chrono::high_resolution_clock::now();
    srand(static_cast<unsigned int>(time(NULL)));
    
    /* create the DigitScanner */
    int nodes[] = {784, 200, 10};
    DigitScanner<float> dgs(nodes, 2, false);
    
    /* files and folders */
    std::string path_data       = "/Users/deiss/Documents/Programmation/C++/DigitScanner/mnist_data/";
    std::string path_folder_ANN = "/Users/deiss/Documents/Programmation/C++/DigitScanner/";
    std::string filename_ANN    = "dgs_params.txt";
    
    /* actions */
    dgs.load(path_folder_ANN, filename_ANN);
    //dgs.train(path_data, 60000, 0, 2, 10, 0.5, 0);
    //dgs.test(path_data, 10000, 0);
    //dgs.save(path_folder_ANN, filename_ANN);
    
    /* Creates a Window for testing */
    Window *w = new Window(280, 280);
    w->setDgs(&dgs);
    w->setSceneWidth(280);
    w->init();
    w->launch();
    
    print_elapsed_time(begin);
    return 0;
}

/* Computes and print execution time */
void print_elapsed_time(chrono_clock begin) {
    chrono_clock end = std::chrono::high_resolution_clock::now();
    auto         dur = end - begin;
    auto         ms  = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
    std::cout << static_cast<double>(ms)/1000 << " s" << std::endl;
}
