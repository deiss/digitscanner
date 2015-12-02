#include <cmath>
#include <thread>

#include "ANN.hpp"

/* ANN constructor */
ANN::ANN(int* nb_nodes, int nb_right_layers)
    : nb_right_layers(nb_right_layers),
      nb_nodes(new int[nb_right_layers+1]),
      input(new ANNLeftLayer(nb_nodes[0])),
      right_layers(new ANNRightLayer*[nb_right_layers]) {
    ANNLayer *previous = input;
    for(int i=0 ; i<nb_right_layers+1 ; i++) this->nb_nodes[i] = nb_nodes[i];
    for(int i=0 ; i<nb_right_layers ; i++) {
        ANNRightLayer *l = new ANNRightLayer(nb_nodes[i+1], previous);
        right_layers[i]  = l;
        previous         = l;
        random_init_values(l);
    }
}

/* ANN desctructor */
ANN::~ANN() {
    delete input;
    for(int i=0 ; i<nb_right_layers ; i++) delete right_layers[i];
    delete [] right_layers;
    delete [] nb_nodes;
}

/* Backpropagation algorithm using the cross-entropy cost function */
ANN::nabla_pair ANN::backpropagation_cross_entropy(const Matrix* training_input, const Matrix* training_output) {
    const Matrix **activations = feedforward_complete(training_input);
    const Matrix **nabla_W     = new const Matrix*[nb_right_layers];
    const Matrix **nabla_B     = new const Matrix*[nb_right_layers];
          Matrix *d            = new Matrix(activations[nb_right_layers]);
    d->operator-(training_output);
    Matrix *at = new Matrix(activations[nb_right_layers-1]); at->transpose();         // aT
    Matrix *nw = new Matrix(d);                              nw = nw->operator*(at);  // d*aT
    nabla_W[nb_right_layers-1] = nw;
    nabla_B[nb_right_layers-1] = d;
    delete at;
    // backward
    for(int i=nb_right_layers-2 ; i>=0 ; i--) {
        const Matrix  *a  = activations[i+1];
              Matrix  *sp = Matrix::Ones(a->getI());                      sp->operator-(a); sp->element_wise_product(a);
              Matrix  *wt  = new Matrix(right_layers[i+1]->getWeights()); wt->transpose();
        d = wt->operator*(d); d->element_wise_product(sp);
        Matrix *at = new Matrix(activations[i]); at->transpose();
        Matrix *nw = new Matrix(d);              nw = nw->operator*(at);
        nabla_W[i] = nw;
        nabla_B[i] = d;
        delete at;
        delete sp;
    }
    for(int i=1 ; i<=nb_right_layers ; i++) delete activations[i];
    delete [] activations; // activations[0]=input is not deleted
    return nabla_pair(nabla_W, nabla_B);
}

/* Feedforward algorithm to be used to compute the output */
const Matrix *ANN::feedforward(const Matrix* X) {
    const Matrix *current = X;
    for(int i=0 ; i<nb_right_layers ; i++) {
        ANNRightLayer *current_layer = right_layers[i];
        Matrix        *W             = current_layer->getWeights();
        Matrix        *B             = current_layer->getBiases();
        Matrix        *a             = new Matrix(W);
        a = a->operator*(current)->operator+(B)->sigmoid();
        if(current!=X) delete current;
        current = a;
    }
    return current;
}

/* Feedforward algorithm to be used in the backpropagation algorithm */
const Matrix **ANN::feedforward_complete(const Matrix* X) {
    const Matrix **activations = new const Matrix*[nb_right_layers+1];
    activations[0]             = X;
    for(int i=0 ; i<nb_right_layers ; i++) {
        ANNRightLayer *current_layer = right_layers[i];
        Matrix        *W             = current_layer->getWeights();
        Matrix        *B             = current_layer->getBiases();
        Matrix        *a             = new Matrix(W);
        a = a->operator*(activations[i])->operator+(B)->sigmoid();
        activations[i+1] = a;
    }
    return activations;
}

/* Initializes the network's weights and biases with a Gaussian generator */
void ANN::random_init_values(ANNRightLayer* l) {
    Matrix *W = l->getWeights();
    Matrix *B = l->getBiases();
    std::default_random_engine       generator;
    std::normal_distribution<double> gauss_biases(0, 1);
    std::normal_distribution<double> gauss_weights(0, 1.0/sqrt(l->getPreviousLayer()->getNbNodes()));
    for(int i = 0 ; i<W->getI() ; i++) {
        for(int j = 0 ; j<W->getJ() ; j++) W->operator()(i, j) = gauss_weights(generator);
        B->operator()(i, 0) = gauss_biases(generator);
    }
}

/* Stochastic Gradient Descent algorithm */
void ANN::SGD(std::vector<const Matrix*>* training_input, std::vector<const Matrix*>* training_output, const int training_set_len, const int nb_epoch, const int batch_len, const double eta, const double alpha) {
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
            threads.push_back(std::thread(&ANN::SGD_batch_update, this, training_input, training_output, &shuffle, training_set_len, batch_counter, batch_len, eta, alpha));
            batch_counter += batch_len;
        }
        for(std::thread& t : threads) t.join();
        std::cout << i << "..." << std::endl;
    }
}

/* Stochastic Gradient Descent algorithm for a batch */
void ANN::SGD_batch_update(std::vector<const Matrix*>* training_input, std::vector<const Matrix*>* training_output, std::map<int, int>* shuffle, const int training_set_len, int batch_counter, const int batch_len, const double eta, const double alpha) {
    std::vector<Matrix *> nabla_W; nabla_W.reserve(nb_right_layers);
    std::vector<Matrix *> nabla_B; nabla_B.reserve(nb_right_layers);
    for(int i=0 ; i<nb_right_layers ; i++) {
        nabla_W.push_back(new Matrix(nb_nodes[i+1], nb_nodes[i]));
        nabla_B.push_back(new Matrix(nb_nodes[i+1], 1));
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

        