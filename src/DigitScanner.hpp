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
This class defines the digit scanner. It has a neural network and functions
to train, test and play with it. This class is a template so that the neural-
network can have several floating point accuracies.
*/

#ifndef DigitScanner_hpp
#define DigitScanner_hpp

#include <vector>

#include "GLUT.hpp"

#include "FNN.hpp"
#include "Matrix.hpp"

template<typename T>
class DigitScanner {

    public:

        typedef std::chrono::time_point<std::chrono::high_resolution_clock> chrono_clock;
    
        DigitScanner();
        DigitScanner(std::vector<int>);
        ~DigitScanner();
    
        void init();
        void set_layers(std::vector<int>);
    
        bool load(std::string);
        bool save(std::string);
        void train(std::string, const int, const int, const int, const int, const double, const double, const int);
        void train_thread(std::string, const int, const int, const int, const int, const double, const double, const int, const int, const int, const int, std::map<int, int>, bool);
        void test(std::string, const int, const int, const int);
    
        void draw(bool);
        void guess();
        void scan(int, int, unsigned char);
        void reset();
    
    private:
    
        std::string create_progress_bar(double);
        double      elapsed_time(chrono_clock);

        FNN<T>*       fnn;     /* feedforward neural network */
        Matrix<float> digit;   /* input digit, 784 pixels of the picture */

};



/*
Initializes the variables.
*/
template<typename T>
DigitScanner<T>::DigitScanner() {
    init();
}

/*
Initializes the variables
*/
template<typename T>
DigitScanner<T>::DigitScanner(std::vector<int> p_layers) :
    fnn(new FNN<T>(p_layers)) {
    init();
}

/*
Frees the memory by deleting the neural network
and the input matrix.
*/
template<typename T>
DigitScanner<T>::~DigitScanner() {
    delete fnn;
    digit.free();
}

/*
Creates the input matrix and fill it with 0.
*/
template<typename T>
void DigitScanner<T>::init() {
    digit.set_dimensions(784, 1);
    for(int i=0 ; i<784 ; i++) digit(i, 0) = 0;
}

/*
Creates the neural network if not done in the constructor.
*/
template<typename T>
void DigitScanner<T>::set_layers(std::vector<int> p_layers) {
    if(fnn) delete fnn;
    fnn = new FNN<T>(p_layers);
}

/*
Draws the digit created by the user. Can draw either the background or
the digit.
*/
template<typename T>
void DigitScanner<T>::draw(bool background) {
    for(int i=0 ; i<28 ; i++) {
        for(int j=0 ; j<28 ; j++) {
            unsigned char color = digit(i*28+j, 0);
            if((background && color==0) || (!background && color>0)) {
                glColor3ub(color, color, color);
                glBegin(GL_QUADS);
                glVertex2d(10*j, 10*(27-i));
                glVertex2d(10*(j+1), 10*(27-i));
                glVertex2d(10*(j+1), 10*(27-i+1));
                glVertex2d(10*j, 10*(27-i+1));
                glEnd();
            }
        }
    }
}

/*
Uses the neural network to guess which number is drawn using
the feedforward function of the neural network. The output is the
output of the neural network that has the highest value.
*/
template<typename T>
void DigitScanner<T>::guess() {
    const Matrix<T> y = fnn->feedforward(&digit);
    int kmax = 0;
    for(int k=0 ; k<10 ; k++) { if(y(k, 0)>y(kmax, 0)) kmax = k; }
    std::cout << "You drew: " << kmax << std::endl;
}

/*
Clears the drawing area.
*/
template<typename T>
void DigitScanner<T>::reset() {
    for(int i=0 ; i<784 ; i++) {
        digit(i, 0) = 0;
    }
}

/*
Stores the number being drawn: if the new pixel's color
is whiter than the previous one, the color is updated.
*/
template<typename T>
void DigitScanner<T>::scan(int i, int j, unsigned char value) {
    if(value>digit(28*i + j, 0)) digit(28*i + j, 0) = value;
}

/*
Loads a Neural Network from a file.
*/
template<typename T>
bool DigitScanner<T>::load(std::string path) {
    std::cerr << "loading FNN... " << std::flush;
    int              nb_layers;
    std::vector<int> layers;
    std::ifstream    file(path);
    if(file) {
        /* number of layers */
        file >> nb_layers;
        layers.reserve(nb_layers);
        /* number of nodes in each layer */
        for(int i=0 ; i<nb_layers ; i++) { int nb_nodes; file >> nb_nodes; layers.push_back(nb_nodes); }
        fnn = new FNN<T>(layers);
        /* weights and biases */
        for(int i=0 ; i<nb_layers-1 ; i++) {
            FNNRightLayer<T>* current = fnn->getRightLayer(i);
            Matrix<T>         W = current->getWeights();
            Matrix<T>         B = current->getBiases();
            /* W - n2 rows and n1 columns if the second layer has n2 nodes */
            /* and the first one has n1 nodes. */
            for(int j=0 ; j<W.get_I() ; j++) {
                for(int k=0 ; k<W.get_J() ; k++) {
                    file >> W(j, k);
                }
            }
            /* B - one line, n2 values */
            for(int j=0 ; j<B.get_I() ; j++) {
                file >> B(j, 0);
            }
        }
        std::cerr << "FNN successfully loaded: " << nb_layers << " layers (";
        for(int i=0 ; i<nb_layers ; i++) {
            std::cerr << layers.at(i);
            if(i<nb_layers-1) std::cerr << ", ";
            else std::cerr << ")" << std::endl;
        }
        file.close();
        return true;
    }
    else {
        std::cerr << "couldn't open file \"" << path << "\"" << std::endl;
        return false;
    }
}

/*
Saves a Neural Network into a file.
*/
template<typename T>
bool DigitScanner<T>::save(std::string path) {
    std::cerr << "saving FNN... " << std::flush;
    std::ofstream file(path);
    if(!file) {
        std::string answer = "";
        std::cerr << "couldn't create file \"" << path << "\": change filename? (y/n): ";
        std::cin >> answer; std::cin.ignore();
        if(answer=="y") {
            std::cerr << "new path: ";
            std::cin >> path; std::cin.ignore();
            file.open(path);
        }
        else {
            return false;
        }
    }
    if(file) {
        /* number of layers */
        file << (fnn->getNbRightLayers()+1) << std::endl;
        /* number of nodes in each */
        for(int i=0 ; i<fnn->getNbRightLayers()+1 ; i++) file << fnn->getLayers()[i] << " ";
        file << std::endl;
        /* weights and biases */
        for(int i=0 ; i<fnn->getNbRightLayers() ; i++) {
            FNNRightLayer<T>* current = fnn->getRightLayer(i);
            Matrix<T>         W       = current->getWeights();
            Matrix<T>         B       = current->getBiases();
            /* W */
            for(int j=0 ; j<W.get_I() ; j++) {
                for(int k=0 ; k<W.get_J() ; k++) {
                    file << W(j, k) << " ";
                }
                file << std::endl;
            }
            /* B */
            for(int j=0 ; j<B.get_I() ; j++) {
                file << B(j, 0) << " ";
            }
            file << std::endl;
        }
        std::cerr << "FNN successfully saved to \"" << path << "\"" << std::endl;
        file.close();
        return true;
    }
    else {
        std::cerr << "couldn't create file \"" << path << "\"" << std::endl;
        return false;
    }
}

/*
Trains a Neural Network using the Stochastic Gradient Descent algorithm.
The whole dataset is shuffled and sliced in groups of ten pictures. For
every batch, the gradient is computed and the matrices are updated using
the backpropagation algorithm. This runs until the whole dataset has been
completed. Depending on the number of epochs, the whole process can be
run more than once.
*/
template<typename T>
void DigitScanner<T>::train(std::string path_data, const int nb_images, const int nb_images_to_skip, const int nb_epoch, const int batch_len, const double eta, const double alpha, const int nb_threads) {
    /* begining */
    chrono_clock begin_training, begin_epoch;
    begin_training = std::chrono::high_resolution_clock::now();
    /* run for each epoch */
    for(int i=0 ; i<nb_epoch ; i++) {
        begin_epoch = std::chrono::high_resolution_clock::now();
        /* shuffle the training set */
        std::map<int, int> shuffle;
        std::vector<int>   indexes;
        for(int j=nb_images_to_skip ; j<nb_images ; j++)   { indexes.push_back(j); }
        for(int j=0 ; j<nb_images-nb_images_to_skip ; j++) {
            int index = rand() % indexes.size();
            shuffle[j] = indexes.at(index);
            indexes.erase(indexes.begin()+index);
        }
        /* launch threads */
        std::vector<std::thread> threads;
        int                      nb_batches             = nb_images/batch_len;
        int                      nb_batches_per_subsets = nb_batches/nb_threads;
        for(int j=0 ; j<nb_threads ; j++) {
            /* first thread shows progress */
            if(j==0) {
                threads.push_back(&DigitScanner<T>::train_thread, this, path_data, nb_images, nb_images_to_skip, nb_epoch, batch_len, eta, alpha, nb_threads, 0, nb_batches_per_subsets*batch_len, i, shuffle);
            }
            /* last thread computes maximum batches available */
            else if(j==nb_threads-1) {
                int nb_batches_available = nb_batches - j*nb_batches_per_subsets;
                threads.push_back(&DigitScanner<T>::train_thread, this, path_data, nb_images, nb_images_to_skip, nb_epoch, batch_len, eta, alpha, nb_threads, j*nb_batches_per_subsets*batch_len, (j*nb_batches_per_subsets + nb_batches_available)*batch_len, i, shuffle);
            }
            /* middle threads compute nb_batches_per_subset batches */
            else {
                threads.push_back(&DigitScanner<T>::train_thread, this, path_data, nb_images, nb_images_to_skip, nb_epoch, batch_len, eta, alpha, nb_threads, j*nb_batches_per_subsets*batch_len, (j+1)*nb_batches_per_subsets*batch_len, i, shuffle);
            }
        }
        /* join all threads */
        for(int j=0 ; j<nb_threads ; j++) {
            threads.at(j).join();
        }
        std::cerr << "\r    epoch " << (i+1) << "/" << nb_epoch << ": completed in " << elapsed_time(begin_epoch) << " s                          " << std::endl;
    }
    std::cerr << "    training completed in " << elapsed_time(begin_training) << " s" << std::endl;
}

template<typename T>
void DigitScanner<T>::train_thread(std::string path_data, const int nb_images, const int nb_images_to_skip, const int nb_epoch, const int batch_len, const double eta, const double alpha, const int nb_threads, const int data_counter_init, const int data_upper_lim, const int epoch, std::map<int, int> shuffle, bool display) {
    std::string            train_images           = path_data + "train-images.idx3-ubyte";
    std::string            train_labels           = path_data + "train-labels.idx1-ubyte";
    const    int           image_len              = 784;
    const    int           label_len              = 1;
    const    int           image_header_len       = 16;
    const    int           label_header_len       = 8;
    unsigned char*         image                  = new unsigned char[image_len];
    unsigned char*         label                  = new unsigned char[label_len];
    int                    image_counter          = data_counter_init;
    int                    nb_batches             = nb_images/batch_len;
    int                    nb_batches_per_subsets = nb_batches/nb_threads;
    chrono_clock           begin_batch            = std::chrono::high_resolution_clock::now();
    std::ifstream          file_images(train_images, std::ifstream::in | std::ifstream::binary);
    std::ifstream          file_labels(train_labels, std::ifstream::in | std::ifstream::binary);
    std::vector<Matrix<T>> batch_input;  batch_input.reserve(batch_len);
    std::vector<Matrix<T>> batch_output; batch_output.reserve(batch_len);
    for(int k=0 ; k<batch_len ; k++) { Matrix<T> m(image_len, 1); batch_input.push_back(m); }
    for(int k=0 ; k<batch_len ; k++) { Matrix<T> m(10, 1);        batch_output.push_back(m); }
    /* variables for progress bar */
    unsigned long int nb_epoch_len = std::to_string(nb_epoch).length();
    unsigned long int this_epo_len = std::to_string(epoch+1).length();
    std::string       begin_spaces = "";
    if(display) {
        for(int j=0 ; j<nb_epoch_len-this_epo_len ; j++) begin_spaces += " ";
        std::cerr << "    epoch " << (epoch+1) << "/" << nb_epoch << ": " << begin_spaces << "[----------]     0 %" << std::flush;
    }
    while(image_counter<data_upper_lim) {
        /* create batch */
        for(int k=0 ; k<batch_len ; k++, image_counter++) {
            /* set cursor in file */
            file_images.seekg(image_header_len + (nb_images_to_skip + shuffle.at(image_counter))*image_len, std::ios_base::beg);
            file_labels.seekg(label_header_len + (nb_images_to_skip + shuffle.at(image_counter))*label_len, std::ios_base::beg);
            /* read an image from the file */
            file_images.read((char*)image, image_len);
            for(int j=0 ; j<image_len ; j++) batch_input.at(k)(j, 0) = static_cast<double>(image[j])/256;
            /* read the label from the data set and create the expected output matrix */
            file_labels.read((char*)label, label_len);
            batch_output.at(k).fill(0);
            batch_output.at(k)(label[0], 0) = 1;
        }
        /* SGD on the batch */
        fnn->SGD_batch_update(batch_input, batch_output, nb_images, batch_len, eta, alpha);
        /* draw progress bar for thread 1 */
        if(display && elapsed_time(begin_batch)>=0.25) {
            double percentage = static_cast<int>(10000*image_counter/static_cast<double>(nb_batches_per_subsets*batch_len))/100.0;
            std::string begin_spaces = "";
            for(int k=0 ; k<nb_epoch_len-this_epo_len ; k++) begin_spaces += " ";
            std::cerr << "\r    epoch " << (epoch+1) << "/" << nb_epoch << ": " << begin_spaces << create_progress_bar(percentage) << percentage << " %";
            if(nb_threads>1) std::cout << " (thread 1/" << nb_threads << ")";
            std::cout << std::flush;
            begin_batch = std::chrono::high_resolution_clock::now();
        }
    }
    for(Matrix<T> m : batch_input)  m.free();
    for(Matrix<T> m : batch_output) m.free();
    delete [] image;
    delete [] label;
    file_images.close();
    file_labels.close();
}

/*
Tests a Neural Network across the MNIST dataset.
*/
template<typename T>
void DigitScanner<T>::test(std::string path_data, const int nb_images, const int nb_images_to_skip, const int nb_threads) {
    std::string    test_images = path_data + "t10k-images.idx3-ubyte";
    std::string    test_labels = path_data + "t10k-labels.idx1-ubyte";
    const    int   image_len        = 784;
    const    int   label_len        = 1;
    const    int   image_header_len = 16;
    const    int   label_header_len = 8;
    /* beginning */
    chrono_clock begin_test = std::chrono::high_resolution_clock::now();
    std::cerr << "testing on " << (nb_images-nb_images_to_skip) << " images:" << std::endl;
    std::cerr << "    testing [----------]     0 %" << std::flush;
    /* skip the first images */
    std::vector<std::thread> threads;
    std::vector<int>         correct_classification(nb_threads, 0);
    int                      nb_images_per_thread = nb_images/nb_threads;
    for(int i=0 ; i<nb_threads ; i++) {
        /* first thread shows progress */
        if(i==0) {
            threads.push_back(std::thread([=, &correct_classification]() mutable {
                /* open the files */
                std::ifstream  file_images(test_images, std::ifstream::in | std::ifstream::binary);
                std::ifstream  file_labels(test_labels, std::ifstream::in | std::ifstream::binary);
                unsigned char* image = new unsigned char[image_len];
                unsigned char* label = new unsigned char[label_len];
                /* set the file cursor */
                file_images.seekg(image_header_len + nb_images_to_skip*image_len, std::ios_base::cur);
                file_labels.seekg(label_header_len + nb_images_to_skip*label_len, std::ios_base::cur);
                /* compute the results */
                Matrix<T>    test_input(image_len, 1);
                chrono_clock begin_sub_test = std::chrono::high_resolution_clock::now();
                for(int j=0 ; j<nb_images_per_thread ; j++) {
                    /* create input matrix */
                    file_images.read((char*)image, image_len);
                    for(int k=0 ; k<image_len ; k++) test_input(k, 0) = static_cast<double>(image[k])/256;
                    /* read output label */
                    file_labels.read((char*)label, label_len);
                    /* compute output */
                    const Matrix<T> y = fnn->feedforward(&test_input);
                    int kmax = 0;
                    for(int k=0 ; k<10 ; k++) { if(y(k, 0)>y(kmax, 0)) kmax = k; }
                    if(kmax==label[0]) correct_classification.at(0)++;
                    /* prints progress bar */
                    if(elapsed_time(begin_sub_test)>=0.25) {
                        double percentage = static_cast<int>(10000*j/static_cast<double>(nb_images_per_thread))/100.0;
                        std::cerr << "\r    testing: " << create_progress_bar(percentage) << percentage << " %";
                        if(nb_threads>1) std::cout << " (thread 1/" << nb_threads << ")";
                        std::cout << std::flush;
                        begin_sub_test = std::chrono::high_resolution_clock::now();
                    }
                 }
                test_input.free();
                delete [] image;
                delete [] label;
                file_images.close();
                file_labels.close();
            }));
        }
        /* last thread tests maximum available pictures */
        else if(i==nb_threads-1) {
            int nb_images_available = nb_images - i*nb_images_per_thread;
            threads.push_back(std::thread([=, &correct_classification](int thread_index) mutable {
                /* open the files */
                std::ifstream  file_images(test_images, std::ifstream::in | std::ifstream::binary);
                std::ifstream  file_labels(test_labels, std::ifstream::in | std::ifstream::binary);
                unsigned char* image = new unsigned char[image_len];
                unsigned char* label = new unsigned char[label_len];
                /* set the file cursor */
                file_images.seekg(image_header_len + (nb_images_to_skip + i*nb_images_per_thread)*image_len, std::ios_base::cur);
                file_labels.seekg(label_header_len + (nb_images_to_skip + i*nb_images_per_thread)*label_len, std::ios_base::cur);
                /* compute the results */
                Matrix<T>    test_input(image_len, 1);
                for(int j=0 ; j<nb_images_available ; j++) {
                    /* create input matrix */
                    file_images.read((char*)image, image_len);
                    for(int k=0 ; k<image_len ; k++) test_input(k, 0) = static_cast<double>(image[k])/256;
                    /* read output label */
                    file_labels.read((char*)label, label_len);
                    /* compute output */
                    const Matrix<T> y = fnn->feedforward(&test_input);
                    int kmax = 0;
                    for(int k=0 ; k<10 ; k++) { if(y(k, 0)>y(kmax, 0)) kmax = k; }
                    if(kmax==label[0]) correct_classification.at(thread_index)++;
                }
                test_input.free();
                delete [] image;
                delete [] label;
                file_images.close();
                file_labels.close();
            }, i));
        }
        /* middle threads */
        else {
            threads.push_back(std::thread([=, &correct_classification](int thread_index) mutable {
                /* open the files */
                std::ifstream  file_images(test_images, std::ifstream::in | std::ifstream::binary);
                std::ifstream  file_labels(test_labels, std::ifstream::in | std::ifstream::binary);
                unsigned char* image = new unsigned char[image_len];
                unsigned char* label = new unsigned char[label_len];
                /* set the file cursor */
                file_images.seekg(image_header_len + (nb_images_to_skip + i*nb_images_per_thread)*image_len, std::ios_base::cur);
                file_labels.seekg(label_header_len + (nb_images_to_skip + i*nb_images_per_thread)*label_len, std::ios_base::cur);
                /* compute the results */
                Matrix<T>    test_input(image_len, 1);
                for(int j=0 ; j<nb_images_per_thread ; j++) {
                    /* create input matrix */
                    file_images.read((char*)image, image_len);
                    for(int k=0 ; k<image_len ; k++) test_input(k, 0) = static_cast<double>(image[k])/256;
                    /* read output label */
                    file_labels.read((char*)label, label_len);
                    /* compute output */
                    const Matrix<T> y = fnn->feedforward(&test_input);
                    int kmax = 0;
                    for(int k=0 ; k<10 ; k++) { if(y(k, 0)>y(kmax, 0)) kmax = k; }
                    if(kmax==label[0]) correct_classification.at(thread_index)++;
                }
                test_input.free();
                delete [] image;
                delete [] label;
                file_images.close();
                file_labels.close();
            }, i));
        }
    }
    /* join all threads */
    for(int i=0 ; i<nb_threads ; i++) {
        threads.at(i).join();
    }
    int correct = 0;
    for(int c : correct_classification) correct += c;
    std::cerr << "\r    testing completed in " << elapsed_time(begin_test) << " s                           " << std::endl;
    std::cerr << "    " << correct << "/" << nb_images << " (" << 100*static_cast<double>(correct)/nb_images << " %) images correctly classified" << std::endl;
}

/*
Creates a textual progress bar.
*/
template<typename T>
std::string DigitScanner<T>::create_progress_bar(double percentage) {
    std::string percentage_str = std::to_string(percentage);
    std::string spaces         = "";
    std::string progress_bar   = "[";
    for(int j=0 ; j<static_cast<int>(percentage/10) ; j++)  progress_bar += "#";
    for(int j=static_cast<int>(percentage/10) ; j<10 ; j++) progress_bar += "-";
    progress_bar += "]";
    for(int j=4 ; j>=0 ; j--) {
        if(percentage_str.at(j)=='0')      { spaces += " "; }
        else if(percentage_str.at(j)=='.') { spaces += " "; break; }
        else                               { break; }
    }
    return progress_bar + " " + spaces;
}

/*
Computes execution time.
*/
template<typename T>
double DigitScanner<T>::elapsed_time(chrono_clock begin) {
    chrono_clock end = std::chrono::high_resolution_clock::now();
    auto         dur = end - begin;
    auto         ms  = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
    return static_cast<int>(ms/10.0)/100.0;
}

#endif
