#ifndef DigitScanner_hpp
#define DigitScanner_hpp

#include "ANN.hpp"

class DigitScanner {

    public:

        DigitScanner(int *, int);
        ~DigitScanner();

        void load(std::string, std::string);
        void save(std::string, std::string);
        void train(std::string, const int, const int, const int, const int, const double, const double);
        void test(std::string, const int, const int);
    
    private:
    
        ANN *ann;

};

#endif /* DigitScanner_hpp */
