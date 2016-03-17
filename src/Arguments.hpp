/*
DigitScanner - Copyright (C) 2016 - Olivier Deiss - olivier.deiss@gmail.com

DigitScanner is a C++ tool to create, train and test feedforward neural
networks (fnn) for handwritten number recognition. The project uses the
MNIST dataset to train and test the neural networks. It is also possible
to draw numbers in a window and ask the tool to guess the number you drew.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef Arguments_hpp
#define Arguments_hpp

#include <iostream>
#include <set>
#include <vector>

class Arguments {

    public:
    
        std::string           fnnin;
        std::string           fnnout;
        std::vector<int>      layers;
        std::string           mnist;
        int                   max_threads;
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
        int  parse_arguments();
        void print_license();

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
