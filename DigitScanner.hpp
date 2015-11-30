#ifndef DigitScanner_hpp
#define DigitScanner_hpp

#include "ANN.hpp"

class DigitScanner {

    public:

        DigitScanner(int *, int);
        ~DigitScanner();

        void load(std::string, std::string);
        void save(std::string, std::string);
        void train(std::string, int, int, int, int, double, double);
        void test(std::string, int, int);
    
    private:
    
        ANN *ann;

};

#endif /* DigitScanner_hpp */
