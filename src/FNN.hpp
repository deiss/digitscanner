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

    typedef std::chrono::time_point<std::chrono::high_resolution_clock> chrono_clock;
    typedef std::pair<std::vector<Matrix<T>>, std::vector<Matrix<T>>>   nabla_pair;

    public:

        FNN(std::vector<int>);
        ~FNN();
    
        int               getNbRightLayers()   const { return nb_right_layers; }
        std::vector<int>  getLayers()          const { return layers; }
        FNNRightLayer<T>* getRightLayer(int i) const { return right_layers[i]; }
    
        void train();
        void use();
    
        const Matrix<T>   feedforward(Matrix<T>*);
        std::vector<Matrix<T>>  feedforward_complete(Matrix<T>*);
        void              random_init_values(FNNRightLayer<T>*);
        void              SGD(std::vector<Matrix<T>>*, std::vector<Matrix<T>>*, const int, const int, const int, const double, const double);
        void              SGD_batch_update(std::vector<Matrix<T>>*, std::vector<Matrix<T>>*, std::map<int, int>*, const int, int, const int, const double, const double);
        nabla_pair        backpropagation_cross_entropy(Matrix<T>&, Matrix<T>&);
    
    private:
    
        double elapsed_time(chrono_clock);
    
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
    
        FNNRightLayer(int nb_nodes, FNNLayer<T>* previous_layer) :
            FNNLayer<T>(nb_nodes),
            previous_layer(previous_layer),
            W(nb_nodes, previous_layer->getNbNodes()),
            B(nb_nodes, 1) {
        }
virtual ~FNNRightLayer() {
        }
    
        FNNLayer<T>* getPreviousLayer() { return previous_layer; }
        Matrix<T>*   getBiases()        { return &B; }
        Matrix<T>*   getWeights()       { return &W; }
    
    private:
    
        FNNLayer<T>* previous_layer;
        Matrix<T>    W;
        Matrix<T>    B;
    
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

This is what is computed by this function, using matrices. Suppose we have
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
//////////////////////////////// nabla_pair (no *) --> memory leak
template<typename T>
typename FNN<T>::nabla_pair FNN<T>::backpropagation_cross_entropy(Matrix<T>& training_input, Matrix<T>& training_output) {
    /* feedforward */
    std::vector<Matrix<T>> activations = feedforward_complete(&training_input);
    /* backpropagation */
    std::vector<Matrix<T>> nabla_CW; nabla_CW.resize(nb_right_layers);
    std::vector<Matrix<T>> nabla_CB; nabla_CB.resize(nb_right_layers);
    Matrix<T> d(activations[nb_right_layers], true);
    Matrix<T> at(activations[nb_right_layers-1], true);
        d -= training_output;
        at.self_transpose();
    Matrix<T> nw(d, true);
        nw *= at;
        at.free();
    nabla_CW[nb_right_layers-1] = nw;
    nabla_CB[nb_right_layers-1] = d;
    /* backward propagation */
    for(int i=nb_right_layers-2 ; i>=0 ; i--) {
        Matrix<T> wt(right_layers[i+1]->getWeights(), true);
            wt.self_transpose();
            d = wt*d;
            wt.free();
            // d = W^t * D
        Matrix<T>* a = &activations[i+1];
        Matrix<T> sp = Matrix<T>::ones_ret_c(a->getI());
            sp -= a;
            sp.self_element_wise_product(a);
            d.self_element_wise_product(sp);
            sp.free();
            // d = [ W^t * D ] ° [ (1-a) ° a ]
        Matrix<T> at(activations[i], true);
            at.self_transpose();
        Matrix<T> nw(d, true);
            nw *= at;
            at.free();
            // nw = [ W^t * D ] ° [ (1-a) ° a ] * a_^t
        nabla_CW[i] = nw;
        nabla_CB[i] = d;
        activations[i+1].free();
        /* activations[0] = input, do not delete */
    }
    
    return nabla_pair(nabla_CW, nabla_CB);
}

/*
Feedforward algorithm to be used to compute the output.
O = WA+B. This function uses the sigmoid function to range
the output in [0 1]. This function is to be called when just
the output is needed.
*/
template<typename T>
const Matrix<T> FNN<T>::feedforward(Matrix<T>* X) {
    std::vector<Matrix<T>> activations;
    activations.push_back(*X);
    for(int i=0 ; i<nb_right_layers ; i++) {
        FNNRightLayer<T>* layer = right_layers[i];
        Matrix<T> a(layer->getWeights(), true);
            a *= activations[i];
            a += layer->getBiases();
            a.self_sigmoid();
            activations.push_back(a);
            if(i>0) activations[i].free();
    }
    return activations[nb_right_layers];
}

/*
Feedforward algorithm to be used in the backpropagation algorithm.
This function is to be called when all the activations are needed,
for instance during the backpropagation step.
*/
template<typename T>
std::vector<Matrix<T>> FNN<T>::feedforward_complete(Matrix<T>* X) {
    std::vector<Matrix<T>> activations;
    activations.push_back(*X);
    for(int i=0 ; i<nb_right_layers ; i++) {
        FNNRightLayer<T>* layer = right_layers[i];
        Matrix<T> a(layer->getWeights(), true);
            a *= activations[i];
            a += layer->getBiases();
            a.self_sigmoid();
            activations.push_back(a);
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
void FNN<T>::SGD(std::vector<Matrix<T>>* training_input, std::vector<Matrix<T>>* training_output, const int training_set_len, const int nb_epoch, const int batch_len, const double eta, const double alpha) {
    chrono_clock begin_training, begin_epoch, begin_batch;
    begin_training = std::chrono::high_resolution_clock::now();
    /* epochs */
    for(int i=0 ; i<nb_epoch ; i++) {
        begin_batch = std::chrono::high_resolution_clock::now();
        std::cout << "shuffling training set..." << std::flush;
        /* shuffle the training data */
        std::map<int, int> shuffle;
        std::vector<int>   indexes;
        for(int j=0 ; j<training_set_len ; j++) { indexes.push_back(j); }
        for(int j=0 ; j<training_set_len ; j++) {
            int index = rand() % indexes.size();
            shuffle[j] = indexes.at(index);
            indexes.erase(indexes.begin()+index);
        }
        std::cout << "\repoch " << (i+1) << "/" << nb_epoch << ": [----------]     0 %" << std::flush;
        /* use all the training dataset */
        int batch_counter = 0;
        begin_epoch = std::chrono::high_resolution_clock::now();
        while(batch_counter<=training_set_len-batch_len) {
            /* SGD on the batch */
            SGD_batch_update(training_input, training_output, &shuffle, training_set_len, batch_counter, batch_len, eta, alpha);
            batch_counter += batch_len;
            if(elapsed_time(begin_batch)>=0.25) {
                double      per          = static_cast<int>(10000*batch_counter/static_cast<double>(training_set_len))/100.0;
                std::string per_str      = std::to_string(per);
                std::string spaces       = "";
                std::string progress_bar = "[";
                for(int i=0 ; i<static_cast<int>(per/10) ; i++)  progress_bar += "#";
                for(int i=static_cast<int>(per/10) ; i<10 ; i++) progress_bar += "-";
                progress_bar += "]";
                for(int i=4 ; i>=0 ; i--) { if(per_str.at(i)=='0' || per_str.at(i)=='.') spaces += " "; else break; }
                std::cout << "\repoch " << (i+1) << "/" << nb_epoch << ": " << progress_bar << " " << spaces << per << " %" << std::flush;
                begin_batch = std::chrono::high_resolution_clock::now();
            }
        }
        std::cout << "\repoch " << (i+1) << "/" << nb_epoch << ": completed in " << elapsed_time(begin_epoch) << " s" << std::endl;
    }
    std::cout << "training completed in " << elapsed_time(begin_training) << " s" << std::endl;
}

/*
Stochastic Gradient Descent algorithm for a batch.
This function is the actual SGD algorithm. It runs the backpropagation
on the whole batch before updating the weights and biases.
*/
template<typename T>
void FNN<T>::SGD_batch_update(std::vector<Matrix<T>>* training_input, std::vector<Matrix<T>>* training_output, std::map<int, int>* shuffle, const int training_set_len, int batch_counter, const int batch_len, const double eta, const double alpha) {
    /* create nabla matrices vectors */
    std::vector<Matrix<T>> nabla_CW;
    std::vector<Matrix<T>> nabla_CB;
    for(int i=0 ; i<nb_right_layers ; i++) {
        nabla_CW.emplace_back(layers[i+1], layers[i]);
        nabla_CB.emplace_back(layers[i+1], 1);
    }
    /* feedforward-backpropagation for each data in the batch and sum the nablas */
    for(int i=0 ; i<batch_len ; i++) {
        nabla_pair delta_nabla = backpropagation_cross_entropy(training_input->at(shuffle->at(batch_counter)), training_output->at(shuffle->at(batch_counter)));
        batch_counter++;
        for(int j=0 ; j<nb_right_layers ; j++) {
            nabla_CW[j] += delta_nabla.first[j];  delta_nabla.first[j].free();
            nabla_CB[j] += delta_nabla.second[j]; delta_nabla.second[j].free();
        }
    }
    /* update the parameters */
    for(int i=0 ; i<nb_right_layers ; i++) {
        nabla_CW[i] *= eta/static_cast<double>(batch_len);
        nabla_CB[i] *= eta/static_cast<double>(batch_len);
        right_layers[i]->getWeights()->operator*=((1-(alpha*eta)/static_cast<double>(training_set_len)));
        right_layers[i]->getWeights()->operator-=(&nabla_CW[i]);
        right_layers[i]->getBiases()->operator-=(&nabla_CB[i]);
        nabla_CW[i].free();
        nabla_CB[i].free();
    }

}

/*
Computes execution time.
*/
template<typename T>
double FNN<T>::elapsed_time(chrono_clock begin) {
    chrono_clock end = std::chrono::high_resolution_clock::now();
    auto         dur = end - begin;
    auto         ms  = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
    return static_cast<int>(ms/10.0)/100.0;
}

#endif
