# DigitScanner

##### Author: DEISS Olivier

## PROJECT

DigitScanner is a tool written in C++ to create, train and test artificial neural networks for handwritten number recognition. The project uses the MNIST dataset to train and test the neural networks. It is also possible to draw numbers in a window and ask the tool to guess the number you drew. The neural network are taught with the Stochastic Gradient Descent algorithm, using the cross-entropy as a cost function.

## LICENSE

This project is licensed under the GPL License. See COPYING for more information.

***

![Screenshot](media/Screenshot.png)

***

### Install

Linux/Mac : 'make' will create the binary in the 'bin' directory'.

### Use

You can start with the existing neural networks in the 'ann' folder and test them with the mnist data_set:

    bin/digitscanner --annin ann/ann_200.txt --test 10000 0 --mnist mnist_data
    
Or you can create a new neural network, with 784 neurons in input, a hidden layer of 100 neurons and an output layer of 10 neurons, and train it twice over the whole dataset with batches of 10 pictures and a learning factor of 0.1. Do not forget to save this neural network with the --annout parameter. The neural network available in ann/ann_100.txt has been created with the following command:

    bin/digitscanner --layers 3 784 100 10 --train 60000 0 2 10 0.1 0 --annout ann_100.txt --mnist mnist_data
    
You can also try to see if adding another hidden layer will improve the test result. It may take a long time, so you can use the --enable_multithreading option to make the process quicker. You can use the --time option to see how long it takes to do the training.

    bin/digitscanner --layers 4 784 100 50 10 --train 60000 0 1 10 0.1 0 --annout ann_100_30.txt --mnist mnist_data --enable_multithreading --time
    
Then you can load the previously created neural networks and test them:

    bin/digitscanner --annin ann_100_30.txt --test 10000 0
    
Or you can use the --gui option to display a window and draw numbers in it. Type 'g' to guess the number and 'r' to reset the drawing area.

    bin/digitscanner --annin ann_100_30.txt --gui

### 

With only one hidden layer, it is possible to achieve significant results on the MNIST testing set, using the 60000 digits from the training set. The MNIST digits are 28x28 black and white pictures. So I used 784 neurons for the input, 10 for the output (1 per digit), and between 30 and 200 neurons for the hidden layer. More neurons in the hidden layer can lead to better performances but also take longer to train.

With 200 neurons in the hidden layer, 14 epochs of training over the whole data set with batches of 10 pictures, up to 98.33% guesses were right on the testing set (10000 pictures). I used a learning rate of 0.1 and no weight decay to get these results. This Neural Network is stored in "dgs_params.txt".

-----------------------------------------------------------------------------------

C++ Functions:
   - 'save':  saves the neural network in a file
   - 'load':  loads a neural network from a file
   - 'train': use training data to update the weights and biases
   - 'test':  gives the output and score for a given testing set

Keys:
   - 'g': guess the number in the drawing area
   - 'r': reset the drawing area



