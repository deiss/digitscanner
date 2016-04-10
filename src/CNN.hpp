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

#ifndef CNN_hpp
#define CNN_hpp

template<typename T> class Feature

template<typename T>
class CNN {

    public:
    
        CNN() {}
        ~CNN() {}
    
    private:
    
        CNNInput<T>*                input;
        CNNConvolutionalLayer<T>**  convolutional_layers;
        CNNMaxPoolingLayer<T>**     max_pooling_layers;
        CNNFullyConnectedLayer<T>** fully_connected_layers;
    
};

template<typename T>
class CNNLayer {

    public:
    
        CNNLayer(int nb_nodes) :
            nb_nodes(nb_nodes) {}
virtual ~CNNLayer() {}
    
};

template<typename T>
class CNNConvolutionalLayer: public CNNLayer {

    public:
    
        CNNConvolutionalLayer();
virtual ~CNNConvolutionalLayer();
    
    private:
    
        

};

template<typename T>
class CNNMaxPoolingLayer: public CNNLayer {

    public:
    
        CNNMaxPoolingLayer();
virtual ~CNNMaxPoolingLayer();
    
    private:
    
    

};

template<typename T>
class CNNFullyConnectedLayer: public CNNLayer {

    public:
    
        CNNFullyConnectedLayer(int p_nb_nodes, CNNLayer<T>* previous_layer) :
            nb_nodes(p_nb_nodes),
            previous_layer(previous_layer),
            W(nb_nodes, previous_layer->get_nb_nodes()),
            B(nb_nodes, 1) {
        }
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

#endif
