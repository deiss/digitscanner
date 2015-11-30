#include <iostream>

#include "DigitScanner.hpp"

void call_from_main();


int main (int argc, char **argv) {

    call_from_main();

    srand(static_cast<unsigned int>(time(NULL)));
    
    int nodes[] = {784, 100, 10};
    DigitScanner dgs(nodes, 2);
    
    std::string  path_data = "/Users/deiss/Documents/Programmation/C++/DigitScanner/mnist_data/";
    
//    dgs.load("/Users/deiss/Documents/Programmation/C++/DigitScanner/", "dgs_params_2.txt");
    dgs.train(path_data, 60000, 0, 1, 5, 0.5, 5);
    dgs.test(path_data, 10000, 0);
    //dgs.save("/Users/deiss/Documents/Programmation/C++/DigitScanner/", "dgs_params_2.txt");
 
    return 0;
    
}