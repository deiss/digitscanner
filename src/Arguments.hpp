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

/*
This class treats the command line arguments and sets their default value.
*/

#ifndef Arguments_hpp
#define Arguments_hpp

#include <iostream>
#include <set>
#include <vector>

class Arguments {

    public:
    
        std::string      fnnin;             /* path of the input neural network file */
        std::string      fnnout;            /* path of the output neural network file */
        std::vector<int> layers;            /* list of number of nodes in each layer */
        std::string      mnist;             /* path to the mnist data_set folder */
        bool             gui;               /* whether the user wants to use the gui */
        int              train_imgnb;       /* number of images to use for training in train mode */
        int              train_imgskip;     /* number of images to skip in train mode */
        int              train_epochs;      /* number of epochs of learning in train mode */
        int              train_batch_len;   /* number of pictures per batch in train mode */
        double           train_eta;         /* learning factor in train mode */
        double           train_alpha;       /* weight decay factor in train mode */
        int              test_imgnb;        /* number of images to test in test mode */
        int              test_imgskip;      /* number of images to skip in test mode */
    
        Arguments(int, char**);
        ~Arguments() {}

        bool is_set(std::string arg) { return arg_set.count(arg); }
        void print_help();
        int  parse_arguments();
        void print_license();

    private:

        bool parse_string_arg(std::string, int*, std::string*, std::string);
        bool check_long_args(std::string);
        bool check_short_args(std::string);
    
        std::set<std::string> arg_set;   /* if an argument is correct and specified it is added to this set */
        int                   argc;      /* number of arguments */
        char**                argv;      /* value of arguments */

};

#endif
