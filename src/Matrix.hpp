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

#ifndef Matrix_hpp
#define Matrix_hpp

template<typename T>
class Matrix {

    public:
    
        Matrix(int, int);
        Matrix(const Matrix&);
        Matrix(const Matrix*);
        ~Matrix();
    
        int getI() const { return I; }
        int getJ() const { return J; }
    
 static Matrix* Ones(int);
 static Matrix  Identity(int);
    
        T       operator()(int, int) const ;
        T&      operator()(int, int);
        Matrix* operator*(T);
        Matrix* operator*(const Matrix*);
        Matrix* operator+(const Matrix*);
        Matrix* operator-(const Matrix*);
    
        void    delete_matrix();
        Matrix* element_wise_product(const Matrix*);
        void    init_matrix();
        void    print() const;
        void    resize(int, int);
 inline T       sigmoid(T) const;
        Matrix* sigmoid();
        Matrix* transpose();
    
    private:

        int I;
        int J;
        T*  matrix;

};



template<typename T>
Matrix<T>::Matrix(int I, int J) : I(I), J(J) {
    init_matrix();
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T>& B) : I(B.I), J(B.J) {
    init_matrix();
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] = B(i, j);
        }
    }
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T>* B) : I(B->I), J(B->J) {
    init_matrix();
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] = B->operator()(i, j);
        }
    }
}

template<typename T>
Matrix<T>::~Matrix() {
    delete_matrix();
}

template<typename T>
T Matrix<T>::sigmoid(T x) const {
    return 1/(1+exp(-x));
}

template<typename T>
Matrix<T>* Matrix<T>::sigmoid() {
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] = sigmoid(matrix[i*J + j]);
        }
    }
    return this;
}

template<typename T>
void Matrix<T>::delete_matrix() {
    delete [] matrix;
}

template<typename T>
void Matrix<T>::init_matrix() {
    matrix = new T[I*J];
    for(int i=0 ; i<I*J ; i++) {
        matrix[i] = 0;
    }
}

template<typename T>
void Matrix<T>::print() const {
    for(int i=0 ; i<I ; i++) {
        std::cout << "| ";
        for(int j=0 ; j<J ; j++) {
            std::cout << matrix[i*J + j] << " ";
        }
        std::cout << "|" << std::endl;
    }
    std::cout << std::endl;
}

template<typename T>
void Matrix<T>::resize(int I, int J) {
    delete_matrix();
    this->I = I;
    this->J = J;
    init_matrix();
}

template<typename T>
Matrix<T>* Matrix<T>::Ones(int I) {
    Matrix* R = new Matrix(I, 1);
    for(int i=0 ; i<I ; i++) R->operator()(i, 0) = 1;
    return R;
}

template<typename T>
Matrix<T> Matrix<T>::Identity(int I) {
    Matrix R(I, I);
    for(int i=0 ; i<I ; i++) R(i, i) = 1;
    return R;
}

template<typename T>
T Matrix<T>::operator()(int i, int j) const {
    return matrix[i*J + j];
}

template<typename T>
T& Matrix<T>::operator()(int i, int j) {
    return matrix[i*J + j];
}

template<typename T>
Matrix<T>* Matrix<T>::operator*(T lambda) {
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] *= lambda;
        }
    }
    return this;
}

/* Matrices product, can only be called on dynamically created (new) Matrix objects */
template<typename T>
Matrix<T>* Matrix<T>::operator*(const Matrix* B) {
    if(B->I!=J) std::cerr << "Matrix dimension dismatch! operator*" << std::endl;
    Matrix* res = new Matrix(I, B->J);
    for(int i=0 ; i<I ; i++) {
        for(int k=0 ; k<B->I ; k++) {
            for(int j=0 ; j<B->J ; j++) {
                res->operator()(i, j) += matrix[i*J + k]*B->operator()(k, j);
            }
        }
    }
    delete this;
    return res;
}

template<typename T>
Matrix<T>* Matrix<T>::operator+(const Matrix* B) {
    if(B->I!=I || B->J!=J) std::cerr << "Matrix dimension dismatch! operator+" << std::endl;
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] += B->operator()(i, j);
        }
    }
    return this;
}

template<typename T>
Matrix<T>* Matrix<T>::operator-(const Matrix* B) {
    if(B->I!=I || B->J!=J) std::cerr << "Matrix dimension dismatch! operator-" << std::endl;
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] -= B->operator()(i, j);
        }
    }
    return this;
}

template<typename T>
Matrix<T>* Matrix<T>::element_wise_product(const Matrix* B) {
    if(B->I!=I || B->J!=J) std::cerr << "Matrix dimension dismatch! element_wise_product" << std::endl;
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] *= B->operator()(i, j);
        }
    }
    return this;
}

template<typename T>
Matrix<T>* Matrix<T>::transpose() {
    Matrix copy = *this;
    resize(J, I);
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] = copy(j, i);
        }
    }
    return this;
}

#endif
