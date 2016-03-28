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

        DigitScanner();
        DigitScanner(std::vector<int>);
        ~DigitScanner();
    
        void init();
        void set_layers(std::vector<int>);
    
        bool load(std::string);
        bool save(std::string);
        void train(std::string, const int, const int, const int, const int, const double, const double);
        void test(std::string, const int, const int);
    
        void draw(bool);
        void guess();
        void scan(int, int, unsigned char);
        void reset();
    
    private:
    
        FNN<T>*        fnn;           /* feedforward neural network */
        Matrix<float>* digit;         /* input digit, 784 pixels of the picture */

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
    delete digit;
}

/*
Creates the input matrix and fill it with 0.
*/
template<typename T>
void DigitScanner<T>::init() {
    digit = new Matrix<float>(784, 1);
    for(int i=0 ; i<784 ; i++) digit->operator()(i, 0) = 0;
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
            unsigned char color = digit->operator()(i*28+j, 0);
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
    const Matrix<T>* y = fnn->feedforward(const_cast<const Matrix<T>*>(digit));
    int kmax = 0;
    for(int k=0 ; k<10 ; k++) { if(y->operator()(k, 0)>y->operator()(kmax, 0)) kmax = k; }
    std::cout << "You drew: " << kmax << std::endl;
    delete y;
}

/*
Clears the drawing area.
*/
template<typename T>
void DigitScanner<T>::reset() {
    for(int i=0 ; i<784 ; i++) {
        digit->operator()(i, 0) = 0;
    }
}

/*
Stores the number being drawn: if the new pixel's color
is whiter than the previous one, the color is updated.
*/
template<typename T>
void DigitScanner<T>::scan(int i, int j, unsigned char value) {
    if(value>digit->operator()(28*i + j, 0)) digit->operator()(28*i + j, 0) = value;
}

/*
Loads a Neural Network from a file.
*/
template<typename T>
bool DigitScanner<T>::load(std::string path) {
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
            Matrix<T>*        W       = current->getWeights();
            Matrix<T>*        B       = current->getBiases();
            /* W - n2 rows and n1 columns if the second layer has n2 nodes */
            /* and the first one has n1 nodes. */
            for(int j=0 ; j<W->getI() ; j++) {
                for(int k=0 ; k<W->getJ() ; k++) {
                    file >> W->operator()(j, k);
                }
            }
            /* B - one line, n2 values */
            for(int j=0 ; j<B->getI() ; j++) {
                file >> B->operator()(j, 0);
            }
        }
        std::cerr << "fnn successfully loaded: " << nb_layers << " layers (";
        for(int i=0 ; i<nb_layers ; i++) {
            std::cout << layers.at(i);
            if(i<nb_layers) std::cout << ", ";
            else std::cout << ")" << std::endl;
        }
        file.close();
        return true;
    }
    else {
        std::cerr << "Couldn't open file \"" << path << "\"." << std::endl;
        return false;
    }
}

/*
Saves a Neural Network into a file.
*/
template<typename T>
bool DigitScanner<T>::save(std::string path) {
    std::ofstream file(path);
    if(file) {
        /* number of layers */
        file << (fnn->getNbRightLayers()+1) << std::endl;
        /* number of nodes in each */
        for(int i=0 ; i<fnn->getNbRightLayers()+1 ; i++) file << fnn->getLayers()[i] << " ";
        file << std::endl;
        /* weights and biases */
        for(int i=0 ; i<fnn->getNbRightLayers() ; i++) {
            FNNRightLayer<T>* current = fnn->getRightLayer(i);
            Matrix<T>*        W       = current->getWeights();
            Matrix<T>*        B       = current->getBiases();
            /* W */
            for(int j=0 ; j<W->getI() ; j++) {
                for(int k=0 ; k<W->getJ() ; k++) {
                    file << W->operator()(j, k) << " ";
                }
                file << std::endl;
            }
            /* B */
            for(int j=0 ; j<B->getI() ; j++) {
                file << B->operator()(j, 0) << " ";
            }
            file << std::endl;
        }
        file.close();
        return true;
    }
    else {
        std::cerr << "Couldn't create file \"" << path << "\"." << std::endl;
        return false;
    }
}

/*
Tests a Neural Network across the MNIST dataset.
*/
template<typename T>
void DigitScanner<T>::test(std::string path_data, const int nb_images, const int nb_images_to_skip) {
    std::string    test_images = path_data + "t10k-images.idx3-ubyte";
    std::string    test_labels = path_data + "t10k-labels.idx1-ubyte";
    std::ifstream  file_images(test_images, std::ifstream::in | std::ifstream::binary);
    std::ifstream  file_labels(test_labels, std::ifstream::in | std::ifstream::binary);
    const    int   image_len        = 784;
    const    int   label_len        = 1;
    const    int   image_header_len = 16;
    const    int   label_header_len = 8;
    unsigned char* image = new unsigned char[image_len];
    unsigned char* label = new unsigned char[label_len];
    /* skip the first images */
    file_images.seekg(image_header_len + nb_images_to_skip*image_len, std::ios_base::cur);
    file_labels.seekg(label_header_len + nb_images_to_skip*label_len, std::ios_base::cur);
    /* compute the results */
    int        right_guesses = 0;
    Matrix<T>* test_input    = new Matrix<T>(image_len, 1);
    for(int i=0 ; i<nb_images ; i++) {
        /* create input matrix */
        file_images.read((char*)image, image_len);
        for(int j=0 ; j<image_len ; j++) test_input->operator()(j, 0) = static_cast<double>(image[j])/256;
        /* read output label */
        file_labels.read((char*)label, label_len);
        /* compute output */
        const Matrix<T>* y = fnn->feedforward(const_cast<const Matrix<T>*>(test_input));
        int kmax = 0;
        for(int k=0 ; k<10 ; k++) { if(y->operator()(k, 0)>y->operator()(kmax, 0)) kmax = k; }
        if(kmax==label[0]) right_guesses++;
        delete y;
    }
    /* displays the score */
    std::cout << 100*static_cast<double>(right_guesses)/nb_images << " %" << std::endl;
    delete test_input;
    delete [] image;
    delete [] label;
    file_images.close();
    file_labels.close();
}

/*
Trains a Neural Network using the Stochastic Gradient Descent algorithm.
*/
template<typename T>
void DigitScanner<T>::train(std::string path_data, const int nb_images, const int nb_images_to_skip, const int nb_epoch, const int batch_len, const double eta, const double alpha) {
    std::string    train_images = path_data + "train-images.idx3-ubyte";
    std::string    train_labels = path_data + "train-labels.idx1-ubyte";
    std::ifstream  file_images(train_images, std::ifstream::in | std::ifstream::binary);
    std::ifstream  file_labels(train_labels, std::ifstream::in | std::ifstream::binary);
    const    int   image_len        = 784;
    const    int   label_len        = 1;
    const    int   image_header_len = 16;
    const    int   label_header_len = 8;
    unsigned char* image            = new unsigned char[image_len];
    unsigned char* label            = new unsigned char[label_len];
    /* skips the first images */
    file_images.seekg(image_header_len + nb_images_to_skip*image_len, std::ios_base::cur);
    file_labels.seekg(label_header_len + nb_images_to_skip*label_len, std::ios_base::cur);
    /* train across the remaning data */
    std::vector<const Matrix<T>*> training_input;  training_input.reserve(nb_images);
    std::vector<const Matrix<T>*> training_output; training_output.reserve(nb_images);
    /* create the training set */
    for(int i=0 ; i<nb_images ; i++) {
        /* read an image from the file */
        Matrix<T>* input = new Matrix<T>(image_len, 1);
        file_images.read((char*)image, image_len);
        for(int j=0 ; j<image_len ; j++) input->operator()(j, 0) = static_cast<double>(image[j])/256;
        training_input.push_back(input);
        /* read the label from the data set and create the expected output matrix */
        Matrix<T>* output = new Matrix<T>(10, 1);
        file_labels.read((char*)label, label_len);
        output->operator()(label[0], 0) = 1;
        training_output.push_back(output);
    }
    /* Stochastic Gradient Descent */
    fnn->SGD(&training_input, &training_output, nb_images, nb_epoch, batch_len, eta, alpha);
    for(const Matrix<T>* m : training_input)  delete m;
    for(const Matrix<T>* m : training_output) delete m;
    delete [] image;
    delete [] label;
    file_images.close();
    file_labels.close();
}

#endif
