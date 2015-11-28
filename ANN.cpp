#include <cmath>

#include "ANN.hpp"

/* ANN constructor */
ANN::ANN(int *nb_nodes, int nb_right_layers)
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

/* Backpropagation algorithm */
ANN::nabla_pair *ANN::backpropagation(const Matrix *training_input, const Matrix *training_output) {
    const Matrix **activations = feedforward_complete(training_input);
    const Matrix **delta       = new const Matrix*[nb_right_layers];
    const Matrix **nabla_W     = new const Matrix*[nb_right_layers];
    const Matrix **nabla_B     = new const Matrix*[nb_right_layers];
    // output error
    const Matrix *a  = activations[nb_right_layers];
          Matrix *o  = Matrix::Ones(a->getI());
          Matrix *d  = new Matrix(a);
          Matrix *sp = new Matrix(o);
    sp->operator-(a)->element_wise_product(a);
    d->operator-(training_output)->element_wise_product(sp);
    delta[nb_right_layers-1] = d;
    delete sp;
    delete o;
    // backward
    for(int i=nb_right_layers-1 ; i>=1 ; i--) {
        const Matrix *a  = activations[i];
              Matrix *o  = Matrix::Ones(a->getI());
              Matrix *d  = new Matrix(right_layers[i]->getWeights());
              Matrix *sp = new Matrix(o);
        sp->operator-(a)->element_wise_product(a);
        d = d->transpose()->operator*(delta[i]);
        d->element_wise_product(sp);
        delta[i-1] = d;
        delete sp;
        delete o;
    }
    for(int i=nb_right_layers-1 ; i>=0 ; i--) {
        Matrix *nw = new Matrix(delta[i]);
        Matrix *at = new Matrix(activations[i]);
        at->transpose();
        nw = nw->operator*(at);
        nabla_W[i] = nw;
        nabla_B[i] = delta[i];
        delete at;
    }
    for(int i=1 ; i<=nb_right_layers ; i++) delete activations[i];
    delete [] activations; // activations[0]=input is not deleted
    delete [] delta;       // the inner objects aren't deleted
    nabla_pair *nabla = new nabla_pair(nabla_W, nabla_B);
    return nabla;
}

/* Feedforward algorithm to be used to compute the output */
const Matrix *ANN::feedforward(const Matrix *X) {
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
const Matrix **ANN::feedforward_complete(const Matrix *X) {
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
void ANN::random_init_values(ANNRightLayer *l) {
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
void ANN::SGD(const Matrix **training_input, const Matrix **training_output, int training_set_len, int nb_epoch, int batch_len, double eta, double alpha) {
    const Matrix **training_input_batch  = new const Matrix*[batch_len];
    const Matrix **training_output_batch = new const Matrix*[batch_len];
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
        int batch_counter = 0;
        while(batch_counter<=training_set_len-batch_len) {
            for(int j=0 ; j<batch_len ; j++) {
                training_input_batch[j]  = training_input[shuffle[batch_counter]];
                training_output_batch[j] = training_output[shuffle[batch_counter]];
                batch_counter++;
            }
            SGD_batch_update(training_input_batch, training_output_batch, batch_len, eta, alpha);
        }
        std::cout << i << "..." << std::endl;
    }
    delete [] training_input_batch;
    delete [] training_output_batch;
}

/* Stochastic Gradient Descent algorithm for a batch */
void ANN::SGD_batch_update(const Matrix **training_input_batch, const Matrix **training_output_batch, int batch_len, double eta, double alpha) {
    Matrix **nabla_W = new Matrix *[nb_right_layers];
    Matrix **nabla_B = new Matrix *[nb_right_layers];
    for(int i=0 ; i<nb_right_layers ; i++) {
        nabla_W[i] = new Matrix(nb_nodes[i+1], nb_nodes[i]);
        nabla_B[i] = new Matrix(nb_nodes[i+1], 1);
    }
    for(int i=0 ; i<batch_len ; i++) {
        std::pair<const Matrix **, const Matrix **> *delta_nabla = backpropagation(training_input_batch[i], training_output_batch[i]);
        for(int j=0 ; j<nb_right_layers ; j++) {
            nabla_W[j]->operator+(delta_nabla->first[j]);
            nabla_B[j]->operator+(delta_nabla->second[j]);
            delete delta_nabla->first[j];
            delete delta_nabla->second[j];
        }
        delete [] delta_nabla->first;
        delete [] delta_nabla->second;
        delete delta_nabla;
    }
    for(int i=0 ; i<nb_right_layers ; i++) {
        nabla_W[i]->operator*(eta/static_cast<double>(batch_len));
        nabla_B[i]->operator*(eta/static_cast<double>(batch_len));
        right_layers[i]->getWeights()->operator-(nabla_W[i]);
        right_layers[i]->getBiases()->operator-(nabla_B[i]);
        delete nabla_W[i];
        delete nabla_B[i];
    }
    delete [] nabla_W;
    delete [] nabla_B;
}

        