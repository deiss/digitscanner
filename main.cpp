#include <iostream>

#include "DigitScanner.hpp"


int main (int argc, char **argv) {

    srand(static_cast<unsigned int>(time(NULL)));
    
    DigitScanner dgs;
    std::string  path_data = "/Users/deiss/Documents/Programmation/C++/DigitScanner/mnist_data/";
    
    //dgs.load("/Users/deiss/Documents/Programmation/C++/DigitScanner/", "dgs_params.txt");
    dgs.train(path_data, 50000, 0, 1, 10, 3, 0);
    dgs.test(path_data, 10000, 50000);
    dgs.save("/Users/deiss/Documents/Programmation/C++/DigitScanner/", "dgs_params.txt");
 
    return 0;
    
}