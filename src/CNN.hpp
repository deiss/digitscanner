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
This class defines a convolutional neural network (cnn) and the associated
methods for initializing, training, and computation of output value.
This class is a template to allow the use of multiple types for the data
stored in the neural network (basically allows float, double, long double).

A neural network is composed of multiple layers. These layers are defined
with the abstract class CNNLayer. The input layer is of type CNNInputLayer,
while the other layers and the output layer are of type CNNFullyConnectedLayer
or CNNMidLayer. CNNMidLayer objects are either convolutional layers or pooling
layers.

                      --------------------------
                      |        CNNLayer        |
                      --------------------------
                        ^          ^          ^
                       /           |           \
                      /            |            \
                     /             |             \
       -----------------    ---------------    --------------------------
       | CNNInputLayer |    | CNNMidLayer |    | CNNFullyConnectedLayer |
       -----------------    ---------------    --------------------------
                               ^      ^
                              /        \
                             /          \
                            /            \
        -------------------------     -------------------
        | CNNConvolutionalLayer |     | CNNPoolingLayer |
        -------------------------     -------------------
        
An CNNInputLayer just has a number of nodes, while a CNNMidLayer or a
CNNFullyConnectedLayer have weight matrices and bias (shared). CNNMidLayer
can either be a CNNConvolutionalLayer or a CNNPoolingLayer. They can be used
in any order between the input layer and he fully connected layers.
*/

#ifndef CNN_hpp
#define CNN_hpp

#include "vector"

#include "Matrix.hpp"

template<typename T> class CNNInputLayer;
template<typename T> class CNNMidLayer;
template<typename T> class CNNConvolutionalLayer;
template<typename T> class CNNPoolingLayer;
template<typename T> class CNNFullyConnectedLayer;

template<typename T>
class CNN {

    public:
    
        CNN() {}
        ~CNN() {}
    
        void compute_activations(Matrix<T>);
        void free_activations();
        //void                   random_init_values(CNNFullyConnectedLayer<T>*);
        //void                   SGD_batch(std::vector<Matrix<T>>, std::vector<Matrix<T>>, const int, const int, const double, const double);
    
    private:
    
        CNNInput<T>*                input;
        CNNMidLayer<T>**            mid_layers;
        CNNFullyConnectedLayer<T>** fully_connected_layers;
    
        int nb_mid_layers;
        int nb_fully_connected_layers;
    
};

template<typename T>
class CNNLayer {

    public:
    
        CNNLayer(int nb_nodes) :
            nb_nodes(nb_nodes) {}
virtual ~CNNLayer() {}

        int get_I()           { return I; }
        int get_J()           { return J; }
        int get_nb_features() { return nb_features; }
    
    protected:
    
        int I;
        int J;
        int nb_features;
    
};

template<typename T>
class CNNInputLayer: public CNNLayer<T> {

    public:
    
        CNNInput(int p_nb_nodes, int p_I, int p_J) :
            nb_features(1),
            I(p_I),
            J(p_J) {}
virtual ~CNNInput() {}

};

template<typename T>
class CNNMidLayer: public CNNLayer<T> {

    public:
    
        CNNMidLayer();
virtual ~CNNMidLayer() {}

virtual void compute_activations(CNNMidLayer*) = 0;
        void free_activations(CNNMidLayer*);

    protected:
    
        std::vector<Matrix<T>> activations;

};

template<typename T>
class CNNConvolutionalLayer: public CNNMidLayer<T> {

    public:
    
        CNNConvolutionalLayer(int p_nb_features, int p_receptive_field_size, int p_stride_length, CNNLayer<T>* p_previous_layer) :
            nb_features(p_nb_features),
            receptive_field_size(p_receptive_field_size),
            stride_length(p_stride_length),
            previous_layer(p_previous_layer),
            I(1 + (previous_layer->I - receptive_field_size)/stride_length),
            J(1 + (previous_layer->J - receptive_field_size)/stride_length) {
            activations.reserve(nb_features);
            W.resize(nb_features); for(int i=0 ; i<nb_features ; i++) W[i].reserve(previous_layer->nb_features);
            B.resize(nb_features); for(int i=0 ; i<nb_features ; i++) B[i].reserve(previous_layer->nb_features);
        }
virtual ~CNNConvolutionalLayer() {}
    
        CNNLayer<T>* get_previous_layer() { return previous_layer; }
        T            get_bias()           { return B; }
        Matrix<T>*   get_weights()        { return &W; }
    
virtual void         compute_activations(CNNMidLayer*);
        void         compute_activations_from_input(Matrix<T>);
    
    private:
    
        int                                 receptive_field_size;
        int                                 stride_length;
        CNNLayer<T>*                        previous_layer;
        std::vector<Matrix<T>>              activations;            /* [this feature map][previous input][i, j] */
        std::vector<std::vector<Matrix<T>>> W;                      /* [this feature map][previous input][i, j] */
        std::vector<std::vector<T>>         B;                      /* [this feature map][previous input] */

};

template<typename T>
class CNNPoolingLayer: public CNNMidLayer<T> {

    public:
    
        CNNPoolingLayer(int p_region_size, int p_stride_length, CNNLayer<T>* p_previous_layer) :
            nb_features(p_previous_layer->nb_features),
            region_size(p_region_size),
            stride_length(p_stride_length),
            previous_layer(p_previous_layer),
            I(1 + (previous_layer->I - receptive_field_size)/stride_length)
            J(1 + (previous_layer->J - receptive_field_size)/stride_length) {}
virtual ~CNNPoolingLayer() {}

virtual void compute_activations(CNNMidLayer*);
    
    private:
    
        int          region_size;
        int          stride_length;
        CNNLayer<T>* previous_layer;

};

template<typename T>
class CNNFullyConnectedLayer: public CNNLayer<T> {

    public:
    
        CNNFullyConnectedLayer(int p_nb_nodes, CNNLayer<T>* p_previous_layer) :
            nb_nodes(p_nb_nodes),
            previous_layer(p_previous_layer),
            W(nb_nodes, previous_layer->get_nb_nodes()),
            B(nb_nodes, 1) {}
virtual ~CNNFullyConnectedLayer() {}
        
        int          get_nb_nodes()       { return nb_nodes; }
        CNNLayer<T>* get_previous_layer() { return previous_layer; }
        Matrix<T>*   get_biases()         { return &B; }
        Matrix<T>*   get_weights()        { return &W; }

    private:
    
        int          nb_nodes;
        CNNLayer<T>* previous_layer;
        Matrix<T>    W;
        Matrix<T>    B;

};



template<typename T>
void CNNMidLayer<T>::free_activations() {
    for(Matrix<T> a : activations) a.free();
    activations.clear();
}

template<typename T>
void CNNConvolutionalLayer<T>::compute_activations_from_input(Matrix<T> input) {
    for(int i=0 ; i<layer.get_nb_features() ; i++) {
        Matrix<T> a(layer.I, layer.J);
        for(int this_layer_i=0 ; this_layer_i<layer.I ; this_layer_i++) {
            for(int this_layer_j=0 ; this_layer_j<layer.J ; this_layer_j++) {
                T var = B[i][0];
                for(int prev_layer_i=0 ; prev_layer_i<receptive_field_size ; prev_layer_i++) {
                    for(int prev_layer_j=0 ; prev_layer_j<receptive_field_size ; prev_layer_j++) {
                        var += W[i][0](prev_layer_i, prev_layer_j)*input(this_layer_i*stride_length + prev_layer_i, this_layer_j*stride_length + prev_layer_j);
                    }
                }
                a(this_layer_i, this_layer_j) = Matrix<T>::sigmoid(var);
            }
        }
        activations.push_back(a);
    }
}

template<typename T>
void CNNConvolutionalLayer<T>::compute_activations(CNNMidLayer* previous_layer) {
    for(int i=0 ; i<nb_features ; i++) {
        Matrix<T> a(I, J);
        for(int this_layer_i=0 ; this_layer_i<layer.I ; this_layer_i++) {
            for(int this_layer_j=0 ; this_layer_j<layer.J ; this_layer_j++) {
                T var = 0;
                for(int j=0 ; j<previous_layer->nb_features ; j++) {
                    var += B[i][j];
                    for(int prev_layer_i=0 ; prev_layer_i<receptive_field_size ; prev_layer_i++) {
                        for(int prev_layer_j=0 ; prev_layer_j<receptive_field_size ; prev_layer_j++) {
                            var += W[i][j](prev_layer_i, prev_layer_j)*input(this_layer_i*stride_length + prev_layer_i, this_layer_j*stride_length + prev_layer_j);
                        }
                    }
                }
                a(this_layer_i, this_layer_j) = Matrix<T>::sigmoid(var);
            }
        }
        activations.push_back(a);
    }
}

template<typename T>
void CNNPoolingLayer<T>::compute_activations(CNNMidLayer* previous_layer) {
    for(int i=0 ; i<nb_features ; i++) {
        Matrix<T> a(I, J);
        for(int this_layer_i=0 ; this_layer_i<I ; this_layer_i++) {
            for(int this_layer_j=0 ; this_layer_j<J ; this_layer_j++) {
                T max_value = -10;
                for(int prev_layer_i=0 ; prev_layer_i<region_size ; prev_layer_i++) {
                    for(int prev_layer_j=0 ; prev_layer_j<region_size ; prev_layer_j++) {
                        T value = previous_layer->activations[i](this_layer_i*stride_length + prev_layer_i, this_layer_j*stride_length + prev_layer_j);
                        if(value>max_value) max_value = value;
                    }
                }
                a(this_layer_i, this_layer_j) = max_value;
            }
        }
        activations.push_back(a);
    }
}

template<typename T>
void CNN<T>::compute_activations(Matrix<T> X) {
    /* input to first convolutional layer */
    FNNMidLayer<T>* layer = mid_layers[0];
    layer->compute_activations_from_input(X);
    /* mid layers */
    for(int i=0 ; i<nb_mid_layers ; i++) {
        mid_layers[i+1]->compute_activations(layer);
        layer = mid_layers[i+1];
    }
    /* fully connected layers */
    for(int i=0 ; i<nb_fully_connected_layers ; i++) {
        /**/
    }
}

template<typename T>
void CNN<T>::free_activations(Matrix<T> X) {
    /* mid layers */
    for(int i=0 ; i<nb_mid_layers ; i++) {
        mid_layers[i]->free_activations();
    }
    /* fully connected layers */
    for(int i=0 ; i<nb_fully_connected_layers ; i++) {
        /**/
    }
}



#endif
