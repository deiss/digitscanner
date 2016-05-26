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

#include <iostream>

#include "DigitScanner.hpp"
#include "Parameters.hpp"
#include "Window.hpp"

void       build_menu(Parameters* const);
const bool check_errors(Parameters* const);

int main(int argc, char **argv) {

    /* args parser */
    Parameters::config p_c {40, 90, 3, 1, 20, 5, 3, 2, Parameters::lang_us};
    Parameters p(argc, argv, p_c);
    build_menu(&p);
    try {
        p.parse_params();
    }
    /* catch errors on parameters */
    catch(const std::exception& e) {
        std::cerr << "error :" << std::endl << "   " << e.what() << std::endl;
        std::cerr << "You can use \"--help\" to get more help." << std::endl;
        return 0;
    }
    /* stops if no arg or help requested */
    if(p.is_spec("help") || argc==1) {
        p.print_help();
        return 0;
    }
    /* or if license is needed */
    else if(p.is_spec("license")) {
        p.print_license();
        return 0;
    }
    /* checks incompatibility among parameters */
    if(!check_errors(&p)) {
        std::cerr << "You can use \"--help\" to get more help." << std::endl;
        return 0;
    }
    
    /* add slash to MNIST folder */
    std::string mnist_folder = "";
    if(p.is_spec("mnist")) {
        mnist_folder = p.str_val("mnist");
        if(mnist_folder.length()>0 && mnist_folder.at(mnist_folder.length()-1)!='/') mnist_folder.push_back('/');
    }
    
    /* initializations */
    srand(static_cast<unsigned int>(time(NULL)));
    
    /* DigitScanner */
    DigitScanner<float> dgs;
    if(p.is_spec("hlayers")) {
        if(p.num_val<int>("hlayers", 1)==0)      dgs.set_layers({784, 10});
        else if(p.num_val<int>("hlayers", 2)==0) dgs.set_layers({784, p.num_val<int>("hlayers", 1), 10});
        else                                     dgs.set_layers({784, p.num_val<int>("hlayers", 1), p.num_val<int>("hlayers", 2), 10});
    }
    else if(p.is_spec("fnnin")) { if(!dgs.load(p.str_val("fnnin"))) return 0; }
    
    /* actions */
    if(p.is_spec("train")) { dgs.train(mnist_folder, p.num_val<int>("train", 1), p.num_val<int>("train", 2), p.num_val<int>("train", 3), p.num_val<int>("train", 4), p.num_val<double>("eta"), p.num_val<double>("alpha"), p.num_val<int>("threads")); }
    if(p.is_spec("test"))  { dgs.test(mnist_folder, p.num_val<int>("test", 1), p.num_val<int>("test", 2), p.num_val<int>("threads")); }

    /* save */
    if(p.is_spec("fnnout")) { dgs.save(p.str_val("fnnout")); }
    
    /* gui */
    if(p.is_spec("gui")) {
        Window *w = new Window(280, 280);
        w->set_dgs(&dgs);
        w->set_scene_width(280);
        w->init();
        w->launch();
    }
    
    return 0;
    
}

void build_menu(Parameters* const p) {
    p->set_program_description("DigitScanner Copyright (C) 2016 Olivier Deiss - olivier.deiss@gmail.com\n\nThis program comes with ABSOLUTELY NO WARRANTY. This is free software, and you are welcome to redistribute it under certain conditions. Type 'digitscanner --license' for details.\n\nDigitScanner uses neural networks to identify handwritten characters from the MNIST dataset.\n\nGithub: https://github.com/CSWest/DigitScanner.git");
    
    p->set_usage("digitscanner [parameters]");

    p->define_param("help", "Displays this help.");
    p->define_param("license", "Displays the GPL license.");
    p->define_num_str_param<std::string>("fnnin", {"path"}, {""}, "Loads a neural network from a file. If not specified, a new neural network is created.");
    p->define_num_str_param<std::string>("fnnout", {"path"}, {""}, "Stores the neural network in a file, at exit. This option is useful when training the neural network. If not specified, the neural network is lost.");
    p->define_num_str_param<int>("hlayers", {"hl1", "hl2"}, {0, 0}, "Creates a neural network with one or two hidden layers and the corresponding number of nodes in each layer. In this command, you only configure the hidden layers. Type 0 for the second hidden layer if you only need one hidden layer. In addition to the specified hidden layers, the first layer (input layer) has 784 nodes, according to the number of pixels in mnist pictures, and the final layer (activation layer) has 10 nodes for the 10 possible digits.");
    p->define_num_str_param<std::string>("mnist", {"path"}, {""}, "Path to the mnist dataset folder.");
    p->define_num_str_param<int>("train", {"imgnb", "imgskip", "epochs", "batch_len"}, {0, 0, 0, 0}, "Trains the neural network with the mnist training set. You can set the number of images to be used for training with <imgnb> (max 60000), the number of images to be skipped at the beggining of the training set with <imgskip>, the number of epochs of training with <epochs>, and the size of the batches with <batch_len>.");
    p->define_num_str_param<double>("eta", {"value"}, {0.5}, "Learning rate. For handwritten number recognition, you can use a value between 0.1 and 1.", true);
    p->define_num_str_param<double>("alpha", {"value"}, {0.1}, "Weight decay factor.", true);
    p->define_num_str_param<int>("test", {"imgnb", "imgskip"}, {0, 0}, "Tests the neural network on the mnist testing set. You can set the number of images to be used for training with <imgnb> (max 10000) and the number of images to be skipped at the beggining of the training set with <imgskip>");
    p->define_param("gui", "Creates a window that enables you to draw numbers. Use 'g' to guess a number and 'r' to reset the drawing area.");
    p->define_num_str_param<int>("threads", {"nb_threads"}, {1}, "Enables multithreading for training or testing.");
}

const bool check_errors(Parameters* const p) {

    /* errors on use of parameters */
    if(!p->is_spec("mnist") && p->is_spec("train"))
        std::cerr << "You cannot train a neural network without specifying the location of the mnist dataset. You can do so with the \"--mnist\" parameter." << std::endl;
    else if(!p->is_spec("mnist") && p->is_spec("test"))
        std::cerr << "You cannot test a neural network without specifying the location of the mnist dataset. You can do so with the \"--mnist\" parameter." << std::endl;
    else if(!p->is_spec("fnnin") && !p->is_spec("hlayers"))
        std::cerr << "You need to either load a neural network from a file with \"--fnnin\" or create a new one with \"--hlayers\"." << std::endl;
    else if(p->is_spec("hlayers") && p->is_spec("fnnin"))
        std::cerr << "You can only either load a neural network from a file or create a new one. Not both." << std::endl;
    else if(p->is_spec("test") && !p->is_spec("fnnin") && !p->is_spec("hlayers"))
        std::cerr << "You cannot test a neural network without loading an existing neural network or creating a new one." << std::endl;
    else if(!p->is_spec("test") && !p->is_spec("train") && !p->is_spec("gui"))
        std::cerr << "Once you create an empty neural network or load an existing one, you need to either train it, test it, or play with it." << std::endl;
    
    /* errors on range */
    else if(p->num_val<int>("threads")<1)
        std::cerr << "You cannot have " << p->num_val<int>("threads") << " thread(s) running. Value should be greater than 1." << std::endl;
    else if(p->num_val<int>("hlayers", 1)<0)
        std::cerr << "The first hidden layer cannot have a negative number of nodes." << std::endl;
    else if(p->num_val<int>("hlayers", 2)<0)
        std::cerr << "The second hidden layer cannot have a negative number of nodes." << std::endl;
    else if(p->num_val<int>("hlayers", 1)==0 && p->num_val<int>("hlayers", 2)>0)
        std::cerr << "You cannot have a second hidden layer with the first one having 0 nodes." << std::endl;
    else if(p->is_spec("train") && (p->num_val<int>("train", 1)>60000))
        std::cerr << "The training set only has 60000 images." << std::endl;
    else if(p->is_spec("train") && (p->num_val<int>("train", 1)+p->num_val<int>("train", 2)>60000))
        std::cerr << "If you skip " << p->num_val<int>("train", 2) << " images, you can only train on " << (60000-p->num_val<int>("train", 2)) << " or less images." << std::endl;
    else if(p->is_spec("test") && (p->num_val<int>("test", 1)>10000))
        std::cerr << "The testing set only has 10000 images." << std::endl;
    else if(p->is_spec("test") && (p->num_val<int>("test", 1)+p->num_val<int>("test", 2)>10000))
        std::cerr << "If you skip " << p->num_val<int>("test", 2) << " images, you can only test on " << (60000-p->num_val<int>("test", 2)) << " or less images." << std::endl;
    else if(p->num_val<double>("eta")<=0)
        std::cerr << "The learning rate cannot be zero or negative." << std::endl;
    else if(p->num_val<double>("alpha")<0)
        std::cerr << "The weght decay cannot be negative." << std::endl;
    
    else
        return true;
    return false;
}











