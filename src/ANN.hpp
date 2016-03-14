/*

Project: DigitScanner
Author: DEISS Olivier

This software is offered under the GPL license. See COPYING for more information.

*/

#ifndef ANN_hpp
#define ANN_hpp

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

template<typename T> class ANNLeftLayer;
template<typename T> class ANNRightLayer;

template<typename T>
class ANN {

    typedef std::pair<const Matrix<T>**, const Matrix<T>**> nabla_pair;

    public:

        ANN(std::vector<int>, int);
        ~ANN();
    
        int               getNbRightLayers()   const { return nb_right_layers; }
        std::vector<int>  getLayers()          const { return layers; }
        ANNRightLayer<T>* getRightLayer(int i) const { return right_layers[i]; }
    
        void train();
        void use();
    
        const Matrix<T>*  feedforward(const Matrix<T>*);
        const Matrix<T>** feedforward_complete(const Matrix<T>*);
        void              random_init_values(ANNRightLayer<T>*);
        void              SGD(std::vector<const Matrix<T>*>*, std::vector<const Matrix<T>*>*, const int, const int, const int, const double, const double);
        void              SGD_batch_update(std::vector<const Matrix<T>*>*, std::vector<const Matrix<T>*>*, std::map<int, int>*, const int, int, const int, const double, const double);
        nabla_pair        backpropagation_cross_entropy(const Matrix<T>*, const Matrix<T>*);
    
    private:
    
        std::vector<int>   layers;
        int                nb_right_layers;
        ANNLeftLayer<T>*   input;
        ANNRightLayer<T>** right_layers;
        int                max_threads;
    
};

template<typename T>
class ANNLayer {

    public:
    
        ANNLayer(int nb_nodes)
            : nb_nodes(nb_nodes) {}
virtual ~ANNLayer() {}
        int getNbNodes() { return nb_nodes; }
    
    protected:
    
        int nb_nodes;
    
};

template<typename T>
class ANNLeftLayer: public ANNLayer<T> {

    public:
    
        ANNLeftLayer(int nb_nodes) : ANNLayer<T>(nb_nodes) {}
virtual ~ANNLeftLayer() {}

};

template<typename T>
class ANNRightLayer: public ANNLayer<T> {

    public:
    
        ANNRightLayer(int nb_nodes, ANNLayer<T>* previous_layer) : ANNLayer<T>(nb_nodes), previous_layer(previous_layer) {
            W = new Matrix<T>(nb_nodes, previous_layer->getNbNodes());
            B = new Matrix<T>(nb_nodes, 1);
        }
virtual ~ANNRightLayer() {
            delete W;
            delete B;
        }
    
        ANNLayer<T>* getPreviousLayer() { return previous_layer; }
        Matrix<T>*   getBiases()        { return B; }
        Matrix<T>*   getWeights()       { return W; }
    
    private:
    
        ANNLayer<T>* previous_layer;
        Matrix<T>*   W;
        Matrix<T>*   B;
    
};



/* ANN constructor. */
template<typename T>
ANN<T>::ANN(std::vector<int> p_layers, int p_max_threads)
    : layers(p_layers),
      nb_right_layers(static_cast<int>(p_layers.size())-1),
      input(new ANNLeftLayer<T>(p_layers[0])),
      right_layers(new ANNRightLayer<T>*[nb_right_layers]),
      max_threads(p_max_threads) {
    ANNLayer<T>* previous = input;
    for(int i=0 ; i<nb_right_layers ; i++) {
        ANNRightLayer<T>* l = new ANNRightLayer<T>(layers[i+1], previous);
        right_layers[i]     = l;
        previous            = l;
        random_init_values(l);
    }
}

/* ANN desctructor. */
template<typename T>
ANN<T>::~ANN() {
    delete input;
    for(int i=0 ; i<nb_right_layers ; i++) delete right_layers[i];
    delete [] right_layers;
}

/* Backpropagation algorithm using the cross-entropy cost function. */
template<typename T>
typename ANN<T>::nabla_pair ANN<T>::backpropagation_cross_entropy(const Matrix<T>* training_input, const Matrix<T>* training_output) {
    const Matrix<T>** activations = feedforward_complete(training_input);
    const Matrix<T>** nabla_W     = new const Matrix<T>*[nb_right_layers];
    const Matrix<T>** nabla_B     = new const Matrix<T>*[nb_right_layers];
          Matrix<T>* d            = new Matrix<T>(activations[nb_right_layers]);
    d->operator-(training_output);
    Matrix<T>* at = new Matrix<T>(activations[nb_right_layers-1]); at->transpose();
    Matrix<T>* nw = new Matrix<T>(d);                              nw = nw->operator*(at);
    nabla_W[nb_right_layers-1] = nw;
    nabla_B[nb_right_layers-1] = d;
    delete at;
    // backward
    for(int i=nb_right_layers-2 ; i>=0 ; i--) {
        const Matrix<T>* a  = activations[i+1];
              Matrix<T>* sp = Matrix<T>::Ones(a->getI());                      sp->operator-(a); sp->element_wise_product(a);
              Matrix<T>* wt  = new Matrix<T>(right_layers[i+1]->getWeights()); wt->transpose();
        d = wt->operator*(d); d->element_wise_product(sp);
        Matrix<T>* at = new Matrix<T>(activations[i]); at->transpose();
        Matrix<T>* nw = new Matrix<T>(d);              nw = nw->operator*(at);
        nabla_W[i] = nw;
        nabla_B[i] = d;
        delete at;
        delete sp;
    }
    for(int i=1 ; i<=nb_right_layers ; i++) delete activations[i];
    delete [] activations; // activations[0]=input is not deleted
    return nabla_pair(nabla_W, nabla_B);
}

/* Feedforward algorithm to be used to compute the output. */
template<typename T>
const Matrix<T>* ANN<T>::feedforward(const Matrix<T>* X) {
    const Matrix<T>* current = X;
    for(int i=0 ; i<nb_right_layers ; i++) {
        ANNRightLayer<T>* current_layer = right_layers[i];
        Matrix<T>*     W                = current_layer->getWeights();
        Matrix<T>*     B                = current_layer->getBiases();
        Matrix<T>*     a                = new Matrix<T>(W);
        a = a->operator*(current)->operator+(B)->sigmoid();
        if(current!=X) delete current;
        current = a;
    }
    return current;
}

/* Feedforward algorithm to be used in the backpropagation algorithm. */
template<typename T>
const Matrix<T>** ANN<T>::feedforward_complete(const Matrix<T>* X) {
    const Matrix<T>** activations = new const Matrix<T>*[nb_right_layers+1];
    activations[0]             = X;
    for(int i=0 ; i<nb_right_layers ; i++) {
        ANNRightLayer<T>*current_layer = right_layers[i];
        Matrix<T>*     W             = current_layer->getWeights();
        Matrix<T>*     B             = current_layer->getBiases();
        Matrix<T>*     a             = new Matrix<T>(W);
        a = a->operator*(activations[i])->operator+(B)->sigmoid();
        activations[i+1] = a;
    }
    return activations;
}

/* Initializes the network's weights and biases with a Gaussian generator. */
template<typename T>
void ANN<T>::random_init_values(ANNRightLayer<T>* l) {
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

/* Stochastic Gradient Descent algorithm. */
template<typename T>
void ANN<T>::SGD(std::vector<const Matrix<T>*>* training_input, std::vector<const Matrix<T>*>* training_output, const int training_set_len, const int nb_epoch, const int batch_len, const double eta, const double alpha) {
    // epoch
    for(int i=0 ; i<nb_epoch ; i++) {
        // shuffle the training data
        std::map<int, int> shuffle;
        std::vector<int>   indexes;
        for(int j=0 ; j<training_set_len ; j++) { indexes.push_back(j); }
        for(int j=0 ; j<training_set_len ; j++) {
            int index = rand() % indexes.size();
            shuffle[j] = indexes.at(index);
            indexes.erase(indexes.begin()+index);
        }
        int                      batch_counter = 0;
        std::vector<std::thread> threads;
        while(batch_counter<=training_set_len-batch_len) {
            int thread_count = 0;
            if(max_threads>1) {
                if(thread_count>max_threads) {
                    threads.at(0).join();
                    threads.erase(threads.begin());
                    thread_count--;
                }
                try{
                    threads.push_back(std::thread(&ANN::SGD_batch_update, this, training_input, training_output, &shuffle, training_set_len, batch_counter, batch_len, eta, alpha));
                    thread_count++;
                }
                catch(std::exception &e) {
                    /* too many threads are already running */
                    for(std::thread& t : threads) t.join();
                    threads.clear();
                    /* try to start this one again */
                    try {
                        threads.push_back(std::thread(&ANN::SGD_batch_update, this, training_input, training_output, &shuffle, training_set_len, batch_counter, batch_len, eta, alpha));
                        thread_count++;
                    }
                    catch(std::exception &e) {
                        std::cerr << "Error while starting a new thread. Exiting." << std::endl;
                        return;
                    }
                }
            }
            else {
                SGD_batch_update(training_input, training_output, &shuffle, training_set_len, batch_counter, batch_len, eta, alpha);
            }
            batch_counter += batch_len;
        }
        if(max_threads>1) { for(std::thread& t : threads) t.join(); }
        std::cerr << "epoch " << (i+1) << " done" << std::endl;
    }
}

/* Stochastic Gradient Descent algorithm for a batch. */
template<typename T>
void ANN<T>::SGD_batch_update(std::vector<const Matrix<T>*>* training_input, std::vector<const Matrix<T>*>* training_output, std::map<int, int>* shuffle, const int training_set_len, int batch_counter, const int batch_len, const double eta, const double alpha) {
    std::vector<Matrix<T>*> nabla_W; nabla_W.reserve(nb_right_layers);
    std::vector<Matrix<T>*> nabla_B; nabla_B.reserve(nb_right_layers);
    for(int i=0 ; i<nb_right_layers ; i++) {
        nabla_W.push_back(new Matrix<T>(layers[i+1], layers[i]));
        nabla_B.push_back(new Matrix<T>(layers[i+1], 1));
    }
    for(int i=0 ; i<batch_len ; i++) {
        nabla_pair delta_nabla = backpropagation_cross_entropy(training_input->at(shuffle->at(batch_counter)), training_output->at(shuffle->at(batch_counter)));
        batch_counter++;
        for(int j=0 ; j<nb_right_layers ; j++) {
            nabla_W[j]->operator+(delta_nabla.first[j]);
            nabla_B[j]->operator+(delta_nabla.second[j]);
            delete delta_nabla.first[j];
            delete delta_nabla.second[j];
        }
        delete [] delta_nabla.first;
        delete [] delta_nabla.second;
    }
    for(int i=0 ; i<nb_right_layers ; i++) {
        nabla_W[i]->operator*(eta/static_cast<double>(batch_len));
        nabla_B[i]->operator*(eta/static_cast<double>(batch_len));
        right_layers[i]->getWeights()->operator*(1-(alpha*eta)/static_cast<double>(training_set_len))->operator-(nabla_W[i]);
        right_layers[i]->getBiases()->operator-(nabla_B[i]);
        delete nabla_W[i];
        delete nabla_B[i];
    }
}

#endif
