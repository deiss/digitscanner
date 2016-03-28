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
This class defines a feedforward neural network (fnn) and the associated
methods for initializing, training, and computation of output value.
This class is a template to allow the use of multiple types for the data
stored in the neural network (basically allows float, double, long double).

A neural network is composed of multiple layers. These layers are defined
with the abstract class FNNLayer. The input layer is of type FNNLeftLayer, 
while the other layers and the output layer are of type FNNRightLayer.

                     ------------
                     | FNNLayer |
                     ------------
                        ^   ^
                       /     \
                      /       \
                     /         \
        ----------------     -----------------
        | FNNLeftLayer |     | FNNRightLayer |
        ----------------     -----------------
        
An FNNLeftLayer just has a number of nodes, while a FNNRightLayer has weight
and bias matrices.
*/

#ifndef FNN_hpp
#define FNN_hpp

#include <cmath>
#include <list>
#include <iostream>
#include <fstream>
#include <map>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include "Matrix.hpp"

template<typename T> class FNNLeftLayer;
template<typename T> class FNNRightLayer;

template<typename T>
class FNN {

    typedef std::pair<const Matrix<T>**, const Matrix<T>**> nabla_pair;

    public:

        FNN(std::vector<int>);
        ~FNN();
    
        int               getNbRightLayers()   const { return nb_right_layers; }
        std::vector<int>  getLayers()          const { return layers; }
        FNNRightLayer<T>* getRightLayer(int i) const { return right_layers[i]; }
    
        void train();
        void use();
    
        const Matrix<T>*  feedforward(const Matrix<T>*);
        const Matrix<T>** feedforward_complete(const Matrix<T>*);
        void              random_init_values(FNNRightLayer<T>*);
        void              SGD(std::vector<const Matrix<T>*>*, std::vector<const Matrix<T>*>*, const int, const int, const int, const double, const double);
        void              SGD_batch_update(std::vector<const Matrix<T>*>*, std::vector<const Matrix<T>*>*, std::map<int, int>*, const int, int, const int, const double, const double);
        nabla_pair        backpropagation_cross_entropy(const Matrix<T>*, const Matrix<T>*);
    
    private:
    
        std::vector<int>   layers;
        int                nb_right_layers;
        FNNLeftLayer<T>*   input;
        FNNRightLayer<T>** right_layers;
    
};

template<typename T>
class FNNLayer {

    public:
    
        FNNLayer(int nb_nodes)
            : nb_nodes(nb_nodes) {}
virtual ~FNNLayer() {}
        int getNbNodes() { return nb_nodes; }
    
    protected:
    
        int nb_nodes;
    
};

template<typename T>
class FNNLeftLayer: public FNNLayer<T> {

    public:
    
        FNNLeftLayer(int nb_nodes) : FNNLayer<T>(nb_nodes) {}
virtual ~FNNLeftLayer() {}

};

template<typename T>
class FNNRightLayer: public FNNLayer<T> {

    public:
    
        FNNRightLayer(int nb_nodes, FNNLayer<T>* previous_layer) : FNNLayer<T>(nb_nodes), previous_layer(previous_layer) {
            W = new Matrix<T>(nb_nodes, previous_layer->getNbNodes());
            B = new Matrix<T>(nb_nodes, 1);
        }
virtual ~FNNRightLayer() {
            delete W;
            delete B;
        }
    
        FNNLayer<T>* getPreviousLayer() { return previous_layer; }
        Matrix<T>*   getBiases()        { return B; }
        Matrix<T>*   getWeights()       { return W; }
    
    private:
    
        FNNLayer<T>* previous_layer;
        Matrix<T>*   W;
        Matrix<T>*   B;
    
};



/*
Initializes the variables and creates the layers according to the
p_layer vector. The layers are linked to each other.
*/
template<typename T>
FNN<T>::FNN(std::vector<int> p_layers) :
    layers(p_layers),
    nb_right_layers(static_cast<int>(p_layers.size())-1),
    input(new FNNLeftLayer<T>(p_layers[0])),
    right_layers(new FNNRightLayer<T>*[nb_right_layers]) {
    FNNLayer<T>* previous = input;
    for(int i=0 ; i<nb_right_layers ; i++) {
        FNNRightLayer<T>* l = new FNNRightLayer<T>(layers[i+1], previous);
        right_layers[i]     = l;
        previous            = l;
        random_init_values(l);
    }
}

/*
Deletes the input, hidden and output layers.
*/
template<typename T>
FNN<T>::~FNN() {
    delete input;
    for(int i=0 ; i<nb_right_layers ; i++) delete right_layers[i];
    delete [] right_layers;
}

/*
Backpropagation algorithm using the cross-entropy cost function. The algorithm
computes the difference between the output for a given set of input and the 
expected output. This gives what should be corrected. This first step is the
feedforward step. The next step is the backpropagation, in which all the weights
and biases differences are computed. This function returns these differences
for one pair of input and output. Usually, the training is done using batches
of input-output data. The parameters are updated only once for the whole batch
of data. Below is the whole mathematical explanation.

The goal of this algorithm is to reduce the cross-entropy cost C defined as

        C = - [ y*ln(a) + (1-y)*ln(1-a) ]
        a = sigmoid(w*a_ + b)                 
    
With a_ the activations of the previous layer. The cross-entropy has the advantage
of enabling the network to learn faster when it is further from the truth, which
is not achievable with the common quadratic cost function 1/2 * ||y-a||^2.

The gradient descent algorithm is an iterative algorithm. At each step, C is
updated by substracting DeltaC:

        DeltaC = NablaC * DeltaX

With Nabl a being the derivative.To make sure DeltaC is always positive so
the cost is effectively reduced at every step, it is possible to choose:

        DeltaX = - n * NablaC
        DeltaC = - n * ||NablaC||^2
        
With n the learning rate. What we will do here is:

        w --> w - n * NablaCw
        b --> b - n * NablaCb
 
So we need to compute NablaCW and NablaCB:

        d = (a-y)
        NablaCw = dC/dw = a_ * d
        NablaCb = dC/db = d
        
The calculus involves sigmoid'(z) = sigmoid(z)*(1-sigmoid(z)), since
sigmoid(z) = 1/(1+e^(−z)).

For the previous layers (x_ means x for the previous layer):

        NablaCw_ = dC/da * da/dz * dz/da_ * da_/dw_
        NablaCb_ = dC/da * da/dz * dz/da_ * da_/db_
        
The calculus gives:

        dC/da * da/dz = (a-y)       // same as first step
        dz/da_ = w                  // a = sig(z)    and  z  = w*a_ + b
        da_/dw_ = a_*(1-a_) * a__   // a_ = sig(z_)  and  z_ = w_*a__ + b_
        da_/db_ = a_*(1-a_)
        
So:

        NablaCw_ = (a-y) * w * a_*(1-a_) * a__ = d * w * a_*(1-a_) * a__
        NablaCb_ = (a-y) * w * a_*(1-a_)       = d * w * a_*(1-a_)

This is what is computed by this function, using matrices Suppose we have
the following 4 layers FNN with 2 hidden layers:

        O
                O       O
        O                       O
                O       O
        O                       O
                O       O
        O
 
        A1      A2      A3      A4        activations for the given layers
            W1      W2      W3            weight matrices between the layers
            B1      B2      B3            bias matrices between the layers

We first compute the nabla for matrices W and B, using the formula:

        D(3)   = A(4) - Y
        NCW(3) = D(3) * A(3)^t
        NCB(3) = D(3)

This is how W3 and B3 needs to be updated. For the previous W and B matrices, 
matrix D is propagated from layer to layer as follow:

        SP   = [ (1) - A(k+1) ] ° A(k+1)   // stands for sigmoid'
        D(k) = [ W(k)^t * D(k+1) ] ° S
        
And then used to compute NCW and NCB:

        NCW(k) = D(k) * A(k)^t
        NCB(k) = D(k)

In these expressions:
        X^t means transpose of X
        (1) means a column of ones of height that of A(k+1).
         °  means an element wise product (Hadamard product)
         *  means a product of matrices
*/
template<typename T>
typename FNN<T>::nabla_pair FNN<T>::backpropagation_cross_entropy(const Matrix<T>* training_input, const Matrix<T>* training_output) {
    /* feedforward */
    const Matrix<T>** activations = feedforward_complete(training_input);
    /* backpropagation */
    const Matrix<T>** nabla_CW = new const Matrix<T>*[nb_right_layers];
    const Matrix<T>** nabla_CB = new const Matrix<T>*[nb_right_layers];
          Matrix<T>*  d        = new Matrix<T>(activations[nb_right_layers]);
    d->operator-(training_output);
    d->print();
    Matrix<T>* at = new Matrix<T>(activations[nb_right_layers-1]); at->transpose();
    Matrix<T>* nw = new Matrix<T>(d);                              nw = nw->operator*(at);
    nabla_CW[nb_right_layers-1] = nw;
    nabla_CB[nb_right_layers-1] = d;
    delete at;
    /* backward propagation */
    for(int i=nb_right_layers-2 ; i>=0 ; i--) {
        const Matrix<T>* a  = activations[i+1];
              Matrix<T>* sp = Matrix<T>::Ones(a->getI());                     sp->operator-(a); sp->element_wise_product(a);
              Matrix<T>* wt = new Matrix<T>(right_layers[i+1]->getWeights()); wt->transpose();
        d = wt->operator*(d); d->element_wise_product(sp);
        Matrix<T>* at = new Matrix<T>(activations[i]); at->transpose();
        Matrix<T>* nw = new Matrix<T>(d);              nw = nw->operator*(at);
        nabla_CW[i] = nw;
        nabla_CB[i] = d;
        delete at;
        delete sp;
    }
    /* do not delete activations[0] (that's the input) */
    for(int i=1 ; i<=nb_right_layers ; i++) delete activations[i];
    delete [] activations;
    return nabla_pair(nabla_CW, nabla_CB);
}

/*
Feedforward algorithm to be used to compute the output.
O = WA+B. This function uses the sigmoid function to range
the output in [0 1]. This function is to be called when just
the output is needed.
*/
template<typename T>
const Matrix<T>* FNN<T>::feedforward(const Matrix<T>* X) {
    const Matrix<T>* current = X;
    for(int i=0 ; i<nb_right_layers ; i++) {
        FNNRightLayer<T>* current_layer = right_layers[i];
        Matrix<T>*        W             = current_layer->getWeights();
        Matrix<T>*        B             = current_layer->getBiases();
        Matrix<T>*        a             = new Matrix<T>(W);
        a = a->operator*(current)->operator+(B)->sigmoid();
        if(current!=X) delete current;
        current = a;
    }
    return current;
}

/*
Feedforward algorithm to be used in the backpropagation algorithm.
This function is to be called when all the activations are needed,
for instance during the backpropagation step.
*/
template<typename T>
const Matrix<T>** FNN<T>::feedforward_complete(const Matrix<T>* X) {
    const Matrix<T>** activations = new const Matrix<T>*[nb_right_layers+1];
    activations[0]             = X;
    for(int i=0 ; i<nb_right_layers ; i++) {
        FNNRightLayer<T>*current_layer = right_layers[i];
        Matrix<T>*       W             = current_layer->getWeights();
        Matrix<T>*       B             = current_layer->getBiases();
        Matrix<T>*       a             = new Matrix<T>(W);
        a = a->operator*(activations[i])->operator+(B)->sigmoid();
        activations[i+1] = a;
    }
    return activations;
}

/*
Initializes the network's weights and biases with a Gaussian generator.
*/
template<typename T>
void FNN<T>::random_init_values(FNNRightLayer<T>* l) {
    Matrix<T>* W = l->getWeights();
    Matrix<T>* B = l->getBiases();
    std::default_random_engine       generator;
    std::normal_distribution<double> gauss_biases(0, 1);
    std::normal_distribution<double> gauss_weights(0, 1.0/sqrt(l->getPreviousLayer()->getNbNodes()));
    for(int i = 0 ; i<W->getI() ; i++) {
        for(int j = 0 ; j<W->getJ() ; j++) W->operator()(i, j) = gauss_weights(generator);
        B->operator()(i, 0) = gauss_biases(generator);
    }
}

/*
Stochastic Gradient Descent algorithm. This function generates multiple
batches of training data, shuffled among the whole training data set, runs
the backpropagation algorithm on this batch, and continues until the whole
data set has been completed. Depending on the number of epochs, the whole
process can be run more than once.
*/
template<typename T>
void FNN<T>::SGD(std::vector<const Matrix<T>*>* training_input, std::vector<const Matrix<T>*>* training_output, const int training_set_len, const int nb_epoch, const int batch_len, const double eta, const double alpha) {
    /* epochs */
    for(int i=0 ; i<nb_epoch ; i++) {
        std::cout << "epoch " << (i+1) << "/" << nb_epoch << " started" << std::endl;
        /* shuffle the training data */
        std::map<int, int> shuffle;
        std::vector<int>   indexes;
        for(int j=0 ; j<training_set_len ; j++) { indexes.push_back(j); }
        for(int j=0 ; j<training_set_len ; j++) {
            int index = rand() % indexes.size();
            shuffle[j] = indexes.at(index);
            indexes.erase(indexes.begin()+index);
        }
        /* use all the training dataset */
        int batch_counter = 0;
        std::chrono::time_point<std::chrono::high_resolution_clock> begin, now;
        begin = std::chrono::high_resolution_clock::now();
        while(batch_counter<=training_set_len-batch_len) {
            /* SGD on the batch */
            SGD_batch_update(training_input, training_output, &shuffle, training_set_len, batch_counter, batch_len, eta, alpha);
            batch_counter += batch_len;
            now      = std::chrono::high_resolution_clock::now();
            auto dur = now - begin;
            auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
            if(ms>=15000) {
                double percentage = static_cast<int>(10000*batch_counter/static_cast<double>(training_set_len))/100.0;
                std::cout << "   epoch " << (i+1) << "/" << nb_epoch << ": " << percentage << " %" << std::endl;
                begin = std::chrono::high_resolution_clock::now();
            }
        }
        std::cout << "epoch " << (i+1) << "/" << nb_epoch << " done" << std::endl;
    }
}

/*
Stochastic Gradient Descent algorithm for a batch.
This function is the actual SGD algorithm. It runs the backpropagation
on the whole batch before updating the weights and biases.
*/
template<typename T>
void FNN<T>::SGD_batch_update(std::vector<const Matrix<T>*>* training_input, std::vector<const Matrix<T>*>* training_output, std::map<int, int>* shuffle, const int training_set_len, int batch_counter, const int batch_len, const double eta, const double alpha) {
    /* create nabla matrices vectors */
    std::vector<Matrix<T>*> nabla_CW; nabla_CW.reserve(nb_right_layers);
    std::vector<Matrix<T>*> nabla_CB; nabla_CB.reserve(nb_right_layers);
    for(int i=0 ; i<nb_right_layers ; i++) {
        nabla_CW.push_back(new Matrix<T>(layers[i+1], layers[i]));
        nabla_CB.push_back(new Matrix<T>(layers[i+1], 1));
    }
    /* feedforward-backpropagation for each data in the batch and sum the nablas */
    for(int i=0 ; i<batch_len ; i++) {
        nabla_pair delta_nabla = backpropagation_cross_entropy(training_input->at(shuffle->at(batch_counter)),
                                                               training_output->at(shuffle->at(batch_counter)));
        batch_counter++;
        for(int j=0 ; j<nb_right_layers ; j++) {
            nabla_CW[j]->operator+(delta_nabla.first[j]);
            nabla_CB[j]->operator+(delta_nabla.second[j]);
            delete delta_nabla.first[j];
            delete delta_nabla.second[j];
        }
        delete [] delta_nabla.first;
        delete [] delta_nabla.second;
    }
    /* update the parameters */
    for(int i=0 ; i<nb_right_layers ; i++) {
        nabla_CW[i]->operator*(eta/static_cast<double>(batch_len));
        nabla_CB[i]->operator*(eta/static_cast<double>(batch_len));
        right_layers[i]->getWeights()->operator*(1-(alpha*eta)/static_cast<double>(training_set_len))->operator-(nabla_CW[i]);
        right_layers[i]->getBiases()->operator-(nabla_CB[i]);
        delete nabla_CW[i];
        delete nabla_CB[i];
    }
}

#endif
