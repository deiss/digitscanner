/*

Project: DigitScanner
Author: DEISS Olivier

This software is offered under the GPL license. See COPYING for more information.

*/

#include "Arguments.hpp"

Arguments::Arguments(int p_argc, char** p_argv) :
    annin(""),
    annout(""),
    mnist(""),
    max_threads(0),
    train_imgnb(0),
    train_imgskip(0),
    train_epochs(0),
    train_batch_len(0),
    train_eta(0),
    train_alpha(0),
    test_imgnb(0),
    test_imgskip(0),
    argc(p_argc),
    argv(p_argv) {
}

void Arguments::print_presentation() {
    std::cerr << "DigitScanner is a tool to create, train and test artificial neural networks for handwritten number recognition." << std::endl;
    std::cerr << std::endl;
}

void Arguments::print_help() {
    std::cerr << "USE: digitscanner [options]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "OPTIONS:" << std::endl;
    std::cerr << "   --help                               Displays this help." << std::endl;
    std::cerr << "   --annin <ann_file_path>              Loads a neural network from a file. If not specified, a new neural network is created." << std::endl;
    std::cerr << "   --annout <ann_file_path>             Stores the neural network in a file, at exit. This option is useful when training the neural network. If not specified, the neural network is lost." << std::endl;
    std::cerr << "   --layers <nb_layers> <1> <2> [...]   Creates a neural network with given number of layers and nodes in each layer. The number of nodes of the first layer has to be set to 784 according to the number of pixels in mnist pictures, and the number of nodes of the right most layer has to be set to 10, for the 10 possible digits." << std::endl;
    std::cerr << "   --mnist <path>                       Path to the mnist dataset folder." << std::endl;
    std::cerr << "   --train <imgnb> <imgskip> <epochs>" << std::endl;
    std::cerr << "           <batch_len> <eta> <alpha>    Trains the neural network with the mnist training set." << std::endl;
    std::cerr << "                                           imgnb:     number of images of the training set to be used for training. Max: 60000." << std::endl;
    std::cerr << "                                           imgskip:   skips the first images of the training set." << std::endl;
    std::cerr << "                                           epochs:    number of epochs in the learning process." << std::endl;
    std::cerr << "                                           batch_len: number of images in a batch." << std::endl;
    std::cerr << "                                           eta:       learning factor. For handwritten number recognition, you can use a value between 0.1 and 1." << std::endl;
    std::cerr << "                                           alpha:     weight decay factor." << std::endl;
    std::cerr << "   --test <imgnb> <imgskip>             Tests the neural network on the mnist testing set." << std::endl;
    std::cerr << "                                           imgnb:   number of images of the testing set to be used for training. Max: 10000." << std::endl;
    std::cerr << "                                           imgskip: skips the first images of the training set." << std::endl;
    std::cerr << "   --time                               Prints the training or testing time." << std::endl;
    std::cerr << "   --gui                                Creates a window that enables you to draw numbers. Commands:" << std::endl;
    std::cerr << "                                           g: using the neural network, guess the number" << std::endl;
    std::cerr << "                                           r: resets the drawing area" << std::endl;
    std::cerr << "   --enable_multithreading <max>        Enables multithreading with a maximum of <max> threads to increase computation performances." << std::endl;
}

int Arguments::parse_arguments() {
    std::string help_msg = "You can use --help to get more help.";
    if(argc==1) {
        std::cerr << "Bad use: you need to tell DigitScanner what to do. You can create or load a neural network, then test it or even play with it." << std::endl;
        std::cerr << help_msg << std::endl;
        return -1;
    }
    else {
        for(int i=1 ; i<argc ; i++) {
            std::string arg_value(argv[i]);
            /* help */
            if(arg_value=="--help") {
                return -2;
            }
            /* string */
            else if(arg_value=="--mnist") {
                if(!parse_string_arg(std::string(argv[i]), &i, &mnist, "You must specify the mnist dataset folder path.\n" + help_msg)) { return -1; }
                if(mnist.at(mnist.length()-1)!='/') mnist.push_back('/');
            }
            else if(arg_value=="--annin") {
                if(!parse_string_arg(std::string(argv[i]), &i, &annin, "You must specify the input neural network file.\n" + help_msg)) { return -1; }
            }
            else if(arg_value=="--annout") {
                if(!parse_string_arg(std::string(argv[i]), &i, &annout, "You must specify the output neural network file.\n" + help_msg)) { return -1; }
            }
            /* integer */
            else if(arg_value=="--enable_multithreading") {
                if(++i<argc) {
                    std::string max_threads_str(argv[i]);
                    try                            { max_threads = std::stoi(max_threads_str); }
                    catch(std::exception const& e) { std::cerr << "The maximum number of threads must be a positive integer." << std::endl; return -1; }
                    if(max_threads<=0) { std::cerr << "The maximum number of threads must be a positive integer." << std::endl; return -1; }
                    else { arg_set.insert("enable_multithreading"); }
                }
                else { std::cerr << "The maximum number of threads is not specified." << std::endl; std::cerr << help_msg << std::endl; return -1; }
            }
            /* commands */
            else if(arg_value=="--train") {
                if(++i<argc) {
                    std::string train_imgnb_str(argv[i]);
                    try                            { train_imgnb = std::stoi(train_imgnb_str); }
                    catch(std::exception const& e) { std::cerr << "The number of images to be used for training must be an integer between 0 and 60000." << std::endl; return -1; }
                    if(train_imgnb<0 || train_imgnb>60000) { std::cerr << "The number of images to be used for training must be an integer between 0 and 60000." << std::endl; return -1; }
                }
                else { std::cerr << "The number of images to be used for training is not specified." << std::endl; std::cerr << help_msg << std::endl; return -1; }
                if(++i<argc) {
                    std::string train_imgskip_str(argv[i]);
                    try                            { train_imgskip = std::stoi(train_imgskip_str); }
                    catch(std::exception const& e) { std::cerr << "The number of images to be skipped must be an integer between 0 and 60000." << std::endl; return -1; }
                    if(train_imgskip<0 || train_imgskip>60000) { std::cerr << "The number of images to be skipped must be an integer between 0 and 60000." << std::endl; return -1; }
                }
                else { std::cerr << "The number of images to be skipped is not specified." << std::endl; std::cerr << help_msg << std::endl; return -1; }
                if(++i<argc) {
                    std::string train_epochs_str(argv[i]);
                    try                            { train_epochs = std::stoi(train_epochs_str); }
                    catch(std::exception const& e) { std::cerr << "The number of epochs must be a positive integer." << std::endl; return -1; }
                    if(train_epochs<=0)            { std::cerr << "The number of epochs is too low." << std::endl; return -1; }
                }
                else { std::cerr << "The number of epochs is not specified." << std::endl; std::cerr << help_msg << std::endl; return -1; }
                if(++i<argc) {
                    std::string train_batch_len_str(argv[i]);
                    try                            { train_batch_len = std::stoi(train_batch_len_str); }
                    catch(std::exception const& e) { std::cerr << "The length of batches must be a positive integer." << std::endl; return -1; }
                    if(train_batch_len<=0)         { std::cerr << "The length of batches is too low." << std::endl; return -1; }
                }
                else { std::cerr << "The length of batches is not specified." << std::endl; std::cerr << help_msg << std::endl; return -1; }
                if(++i<argc) {
                    std::string train_eta_str(argv[i]);
                    try                            { train_eta = std::stod(train_eta_str); }
                    catch(std::exception const& e) { std::cerr << "Eta must be a positive float." << std::endl; return -1; }
                    if(train_eta<=0)               { std::cerr << "Eta is too low." << std::endl; return -1; }
                }
                else { std::cerr << "Eta is not specified." << std::endl; std::cerr << help_msg << std::endl; return -1; }
                if(++i<argc) {
                    std::string train_alpha_str(argv[i]);
                    try                            { train_alpha = std::stod(train_alpha_str); }
                    catch(std::exception const& e) { std::cerr << "Alpha must be a positive float." << std::endl; return -1; }
                    if(train_alpha<0)              { std::cerr << "Alpha must be a positive float." << std::endl; return -1; }
                    else                           { arg_set.insert("train"); }
                }
                else { std::cerr << "Alpha is not specified." << std::endl; std::cerr << help_msg << std::endl; return -1; }
            }
            else if(arg_value=="--test") {
                if(++i<argc) {
                    std::string test_imgnb_str(argv[i]);
                    try                            { test_imgnb = std::stoi(test_imgnb_str); }
                    catch(std::exception const& e) { std::cerr << "The number of images to be used for testing must be an integer between 0 and 10000." << std::endl; return -1; }
                    if(test_imgnb<0 || test_imgnb>10000) { std::cerr << "The number of images to be used for testing must be an integer between 0 and 10000." << std::endl; return -1; }
                }
                else { std::cerr << "The number of images to be used for testing is not specified." << std::endl; std::cerr << help_msg << std::endl; return -1; }
                if(++i<argc) {
                    std::string test_imgskip_str(argv[i]);
                    try                            { test_imgskip = std::stoi(test_imgskip_str); }
                    catch(std::exception const& e) { std::cerr << "The number of images to be skipped must be an integer between 0 and 60000." << std::endl; return -1; }
                    if(test_imgskip<0 || test_imgskip>10000) { std::cerr << "The number of images to be skipped must be an integer between 0 and 60000." << std::endl; return -1; }
                    else { arg_set.insert("test"); }
                }
                else { std::cerr << "The number of images to be skipped is not specified." << std::endl; std::cerr << help_msg << std::endl; return -1; }
            }
            else if(arg_value=="--layers") {
                if(++i<argc) {
                    std::string nb_layers_str(argv[i]);
                    int         nb_layers;
                    try                            { nb_layers = std::stoi(nb_layers_str); }
                    catch(std::exception const& e) { std::cerr << "The number of layers must be a positive integer." << std::endl; return -1; }
                    if(nb_layers<0 || nb_layers>10000) { std::cerr << "The number of layers must be a positive integer." << std::endl; return -1; }
                    else {
                        layers.reserve(nb_layers);
                        for(int j=0 ; j<nb_layers ; j++) {
                            if(++i<argc) {
                                std::string nb_nodes_str(argv[i]);
                                int         nb_nodes;
                                try                            { nb_nodes = std::stoi(nb_nodes_str); }
                                catch(std::exception const& e) { std::cerr << "The number of nodes must be a strictly positive integer." << std::endl; return -1; }
                                if(nb_nodes<1) { std::cerr << "The number of nodes must be a strictly positive integer." << std::endl; return -1; }
                                else { layers.push_back(nb_nodes); }
                            }
                            else {
                                std::cerr << "The number of nodes for the " << j << "th layer is not specified." << std::endl;
                                return -1;
                            }
                        }
                        arg_set.insert("layers");
                    }
                }
                else {
                    std::cerr << "The number of layers is not specified." << std::endl;
                    std::cerr << help_msg << std::endl;
                    return -1;
                }
            }
            /* options */
            else if(arg_value=="--time") {
                arg_set.insert("time");
            }
            else if(arg_value=="--gui") {
                arg_set.insert("gui");
            }
            else {
                std::cerr << "Unknown \"" << arg_value << "\" parameter." << std::endl;
                std::cerr << help_msg << std::endl; return -1;
            }
        }
        /* errors */
        if(!check_long_args(help_msg)) return -1;
    }
    return 0;
}

bool Arguments::parse_string_arg(std::string arg_value, int* i, std::string* arg_container, std::string error_msg) {
    if(++*i<argc) {
        *arg_container = std::string(argv[*i]);
        arg_set.insert(arg_value.substr(2, arg_value.size()-2));
        return true;
    }
    else {
        std::cerr << error_msg << std::endl;
        return false;
    }
}

bool Arguments::parse_short_args(char arg_value, std::string* arg_container, std::string help_msg) {
    if(*arg_container=="") {
        *arg_container = std::string(1, arg_value);
        arg_set.insert(std::string(1, arg_value));
        return true;
    }
    else {
        std::cerr << "You cannot specify both \"-" << *arg_container << "\" and \"-" << arg_value << "\" options." << std::endl;
        std::cerr << help_msg << std::endl;
        return false;
    }
}

bool Arguments::check_long_args(std::string help_msg) {
    if(!arg_set.count("mnist") && arg_set.count("train")) {
        std::cerr << "You cannot train a neural network without specifying the location of the mnist dataset. You can do so by using the --mnist parameter." << std::endl;
        std::cerr << help_msg << std::endl;
        return false;
    }
    else if(!arg_set.count("mnist") && arg_set.count("test")) {
        std::cerr << "You cannot test a neural network without specifying the location of the mnist dataset. You can do so by using the --mnist parameter." << std::endl;
        std::cerr << help_msg << std::endl;
        return false;
    }
    else if(!arg_set.count("annin") && !arg_set.count("layers")) {
        std::cerr << "You need to either load a neural network from a file using --annin or create a new one using --layers." << std::endl;
        std::cerr << help_msg << std::endl;
        return false;
    }
    else if(arg_set.count("layers") && arg_set.count("annin")) {
        std::cerr << "You can only either load a neural network from a file or create a new one using --layers. Not both." << std::endl;
        std::cerr << help_msg << std::endl;
        return false;
    }
    else if(arg_set.count("test") && !arg_set.count("annin") && !arg_set.count("layers")) {
        std::cerr << "You cannot test a neural network without loading an existing neural network or creating a new one." << std::endl;
        std::cerr << help_msg << std::endl;
        return false;
    }
    else if(arg_set.count("test") && arg_set.count("train")) {
        std::cerr << "You can either train or test a neural network." << std::endl;
        std::cerr << help_msg << std::endl;
        return false;
    }
    else if(arg_set.count("layers") && layers.at(0)!=784) {
        std::cerr << "The number of nodes of the first layer has to be set to 784 according to the number of pixels in mnist pictures." << std::endl;
        std::cerr << help_msg << std::endl;
        return false;
    }
    else if(arg_set.count("layers") && layers.at(layers.size()-1)!=10) {
        std::cerr << "The number of nodes of the right most layer has to be set to 10, for the 10 possible digits." << std::endl;
        std::cerr << help_msg << std::endl;
        return false;
    }
    else if(!arg_set.count("test") && !arg_set.count("train") && !arg_set.count("gui")) {
        std::cerr << "Think green! You cannot just create an empty neural network or load an existing one if you do not use it. You need to either train it, test it, or play with it." << std::endl;
        std::cerr << help_msg << std::endl;
        return false;
    }
    return true;
}
