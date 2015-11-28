#ifndef ANN_hpp
#define ANN_hpp

#include <cmath>
#include <list>
#include <iostream>
#include <fstream>
#include <map>
#include <random>
#include <utility>

#include "Matrix.hpp"

class ANNLeftLayer;
class ANNRightLayer;

class ANN {

    typedef std::pair<const Matrix **, const Matrix **> nabla_pair;

    public:

        ANN(int *, int);
        ~ANN();
    
        int            getNbRightLayers()   const { return nb_right_layers; }
        int           *getNbNodes()         const { return nb_nodes; }
        ANNRightLayer *getRightLayer(int i) const { return right_layers[i]; }
    
        void train();
        void use();
    
        const Matrix  *feedforward(const Matrix *);
        const Matrix **feedforward_complete(const Matrix *);
        void           random_init_values(ANNRightLayer*);
        void           SGD(const Matrix **, const Matrix **, int, int, int, double, double);
        void           SGD_batch_update(const Matrix **, const Matrix **, int, double, double);
        nabla_pair    *backpropagation(const Matrix *, const Matrix *);
    
    private:
    
        ANNLeftLayer   *input;
        int            *nb_nodes;
        int             nb_right_layers;
        ANNRightLayer **right_layers;
    
};

class ANNLayer {

    public:
    
        ANNLayer(int nb_nodes)
            : nb_nodes(nb_nodes) {}
virtual ~ANNLayer() {}
        int getNbNodes() { return nb_nodes; }
    
    protected:
    
        int nb_nodes;
    
};

class ANNLeftLayer: public ANNLayer {

    public:
    
        ANNLeftLayer(int nb_nodes) : ANNLayer(nb_nodes) {}
virtual ~ANNLeftLayer() {}

};

class ANNRightLayer: public ANNLayer {

    public:
    
        ANNRightLayer(int nb_nodes, ANNLayer *previous_layer) : ANNLayer(nb_nodes), previous_layer(previous_layer) {
            W = new Matrix(nb_nodes, previous_layer->getNbNodes());
            B = new Matrix(nb_nodes, 1);
        }
virtual ~ANNRightLayer() {
            delete W;
            delete B;
        }
    
        ANNLayer *getPreviousLayer() { return previous_layer; }
        Matrix *getBiases()  { return B; }
        Matrix *getWeights() { return W; }
    
    private:
    
        ANNLayer *previous_layer;
        Matrix   *W;
        Matrix   *B;
    
};

#endif



