#ifndef Arguments_hpp
#define Arguments_hpp

#include <iostream>
#include <set>
#include <vector>

class Arguments {

    public:
    
        std::string           annin;
        std::string           annout;
        std::vector<int>      layers;
        std::string           mnist;
        bool                  time;
        bool                  gui;
        int                   train_imgnb;
        int                   train_imgskip;
        int                   train_epochs;
        int                   train_batch_len;
        double                train_eta;
        double                train_alpha;
        int                   test_imgnb;
        int                   test_imgskip;
    
        Arguments(int, char**);
        ~Arguments() {}

        bool is_set(std::string arg) { return arg_set.count(arg); }
        void print_help();
        void print_presentation();
        int  parse_arguments();

    private:

        bool parse_short_args(char, std::string*, std::string);
        bool parse_string_arg(std::string, int*, std::string*, std::string);
        bool check_long_args(std::string);
        bool check_short_args(std::string);
    
        std::set<std::string> arg_set;
        int                   argc;
        char**                argv;

};

#endif
