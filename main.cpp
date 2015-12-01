#include <iostream>

#include "DigitScanner.hpp"

void call_from_main();


int main(int argc, char **argv) {

    call_from_main();

    srand(static_cast<unsigned int>(time(NULL)));
    
    int nodes[] = {784, 10, 10};
    DigitScanner dgs(nodes, 2);
    
    std::string  path_data = "/Users/deiss/Documents/Programmation/C++/DigitScanner/mnist_data/";
    
   // dgs.load("/Users/deiss/Documents/Programmation/C++/DigitScanner/", "dgs_params_800.txt");
    dgs.train(path_data, 60000, 0, 1, 10, 0.5, 0);
    dgs.test(path_data, 10000, 0);
   // dgs.save("/Users/deiss/Documents/Programmation/C++/DigitScanner/", "dgs_params_800.txt");
 
    return 0;
    
}