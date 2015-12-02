#include <iostream>

#include "DigitScanner.hpp"

void call_from_main();


int main(int argc, char **argv) {

   // call_from_main();

    auto begin = std::chrono::high_resolution_clock::now();
    
    
    

    srand(static_cast<unsigned int>(time(NULL)));
    
    int nodes[] = {784, 200, 10};
    DigitScanner<float> dgs(nodes, 2, false);
    
    std::string  path_data = "/Users/deiss/Documents/Programmation/C++/DigitScanner/mnist_data/";
    
   // dgs.load("/Users/deiss/Documents/Programmation/C++/DigitScanner/", "dgs_params_800.txt");
    dgs.train(path_data, 60000, 0, 10, 10, 0.5, 0);
    dgs.test(path_data, 10000, 0);
   // dgs.save("/Users/deiss/Documents/Programmation/C++/DigitScanner/", "dgs_params_800.txt");
    
    
    
    
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = end - begin;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
    std::cout << static_cast<double>(ms)/1000 << " s" << std::endl;
 
    return 0;
    
}