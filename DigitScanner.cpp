#include "DigitScanner.hpp"

DigitScanner::DigitScanner(int *nodes, int nb_right_layers)
    : ann(new ANN(nodes, nb_right_layers)) {
}

DigitScanner::~DigitScanner() {
    delete ann;
}

void DigitScanner::load(std::string folder, std::string filename) {
    if(ann) delete  ann;
    int             nb_layers;
    int            *nb_nodes;
    std::string     path = folder + filename;
    std::ifstream   file(path);
    // number of layers
    file >> nb_layers;
    nb_nodes = new int[nb_layers];
    // number of nodes in layer
    for(int i=0 ; i<nb_layers ; i++) file >> nb_nodes[i];
    ann = new ANN(nb_nodes, nb_layers-1);
    // weights and biases
    for(int i=0 ; i<nb_layers-1 ; i++) {
        ANNRightLayer *current = ann->getRightLayer(i);
        Matrix        *W = current->getWeights();
        Matrix        *B = current->getBiases();
        // write W
        for(int j=0 ; j<W->getI() ; j++) {
            for(int k=0 ; k<W->getJ() ; k++) {
                file >> W->operator()(j, k);
            }
        }
        // write B inline
        for(int j=0 ; j<B->getI() ; j++) {
            file >> B->operator()(j, 0);
        }
    }
    file.close();
    delete nb_nodes;
}

void DigitScanner::save(std::string folder, std::string filename) {
    std::string   path = folder + filename;
    std::ofstream file(path);
    // number of layers
    file << (ann->getNbRightLayers()+1) << std::endl;
    // number of nodes in each
    for(int i=0 ; i<ann->getNbRightLayers()+1 ; i++) file << ann->getNbNodes()[i] << " ";
    file << std::endl;
    // weights and biases
    for(int i=0 ; i<ann->getNbRightLayers() ; i++) {
        ANNRightLayer *current = ann->getRightLayer(i);
        Matrix        *W = current->getWeights();
        Matrix        *B = current->getBiases();
        // write W
        for(int j=0 ; j<W->getI() ; j++) {
            for(int k=0 ; k<W->getJ() ; k++) {
                file << W->operator()(j, k) << " ";
            }
            file << std::endl;
        }
        // write B inline
        for(int j=0 ; j<B->getI() ; j++) {
            file << B->operator()(j, 0) << " ";
        }
        file << std::endl;
    }
    file.close();
}

void DigitScanner::test(std::string path_data, const int nb_images, const int nb_images_to_skip) {
    // training and test data file path
    std::string    test_images = path_data + "t10k-images.idx3-ubyte";
    std::string    test_labels = path_data + "t10k-labels.idx1-ubyte";
    //std::string    test_images = path_data + "train-images.idx3-ubyte";
    //std::string    test_labels = path_data + "train-labels.idx1-ubyte";
    std::ifstream  file_images(test_images, std::ifstream::in | std::ifstream::binary);
    std::ifstream  file_labels(test_labels, std::ifstream::in | std::ifstream::binary);
    const int      image_len = 784;
    const int      label_len = 1;
    unsigned char *image = new unsigned char[image_len];
    unsigned char *label = new unsigned char[label_len];
    file_images.read((char *)image, 16);
    file_labels.read((char *)label, 8);
    
    // offset
    for(int i=0 ; i<nb_images_to_skip ; i++) {
        file_images.read((char *)image, image_len);
        file_labels.read((char *)label, label_len);
    }
    
    // compute the results
    int     right_guesses = 0;
    Matrix *test_input    = new Matrix(image_len, 1);
    for(int i=0 ; i<nb_images ; i++) {
        // create input matrix
        file_images.read((char *)image, image_len);
        for(int j=0 ; j<image_len ; j++) test_input->operator()(j, 0) = double(image[j])/256;
        
        // read output label
        file_labels.read((char *)label, label_len);
        
        // compute output
        const Matrix *y = ann->feedforward(const_cast<const Matrix *>(test_input));
        int kmax = 0;
        for(int k=0 ; k<10 ; k++) { if(y->operator()(k, 0)>y->operator()(kmax, 0)) kmax = k; }
        if(kmax==label[0]) right_guesses++;
        delete y;
    }
    // print the score
    std::cout << 100*double(right_guesses)/nb_images << " %" << std::endl;
    delete test_input;
    delete [] image;
    delete [] label;
    file_images.close();
    file_labels.close();
}

void DigitScanner::train(std::string path_data, const int nb_images, const int nb_images_to_skip, const int nb_epoch, const int batch_len, const double eta, const double alpha) {
    // training and test data file path
    std::string    train_images = path_data + "train-images.idx3-ubyte";
    std::string    train_labels = path_data + "train-labels.idx1-ubyte";
    std::ifstream  file_images(train_images, std::ifstream::in | std::ifstream::binary);
    std::ifstream  file_labels(train_labels, std::ifstream::in | std::ifstream::binary);
    const    int   image_len = 784;
    const    int   label_len = 1;
    unsigned char *image = new unsigned char[image_len];
    unsigned char *label = new unsigned char[label_len];
    file_images.read((char*)image, 16);
    file_labels.read((char*)label, 8);
    // offset
    for(int i=0 ; i<nb_images_to_skip ; i++) {
        file_images.read((char*)image, image_len);
        file_labels.read((char*)label, label_len);
    }
    // train across all the data set
    std::vector<const Matrix *> training_input;  training_input.reserve(nb_images);
    std::vector<const Matrix *> training_output; training_output.reserve(nb_images);
    // create the training set
    for(int i=0 ; i<nb_images ; i++) {
        // read an image from the file
        Matrix *input = new Matrix(image_len, 1);
        file_images.read((char*)image, image_len);
        for(int j=0 ; j<image_len ; j++) input->operator()(j, 0) = double(image[j])/256;
        training_input.push_back(input);
        // read the label from the data set and create the expected output matrix
        Matrix *output = new Matrix(10, 1);
        file_labels.read((char*)label, label_len);
        output->operator()(label[0], 0) = 1;
        training_output.push_back(output);
    }
    // Stochastic Gradient Descent
    ann->SGD(&training_input, &training_output, nb_images, nb_epoch, batch_len, eta, alpha);
    // clean up
    for(const Matrix* m : training_input)  delete m;
    for(const Matrix* m : training_output) delete m;
    delete [] image;
    delete [] label;
    file_images.close();
    file_labels.close();
}

