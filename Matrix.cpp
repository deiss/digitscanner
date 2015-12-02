#include <cmath>
#include <iostream>

#include "Matrix.hpp"

Matrix::Matrix(int I, int J) : I(I), J(J) {
    init_matrix();
}

Matrix::Matrix(const Matrix& B) : I(B.I), J(B.J) {
    init_matrix();
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] = B(i, j);
        }
    }
}

Matrix::Matrix(const Matrix *B) : I(B->I), J(B->J) {
    init_matrix();
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] = B->operator()(i, j);
        }
    }
}

Matrix::~Matrix() {
    delete_matrix();
}

float Matrix::sigmoid(float x) const {
    return 1/(1+exp(-x));
}

Matrix* Matrix::sigmoid() {
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] = sigmoid(matrix[i*J + j]);
        }
    }
    return this;
}

void Matrix::delete_matrix() {
    delete [] matrix;
}

void Matrix::init_matrix() {
    matrix = new float[I*J];
    for(int i=0 ; i<I*J ; i++) {
        matrix[i] = 0;
    }
}

void Matrix::print() const {
    for(int i=0 ; i<I ; i++) {
        std::cout << "| ";
        for(int j=0 ; j<J ; j++) {
            std::cout << matrix[i*J + j] << " ";
        }
        std::cout << "|" << std::endl;
    }
    std::cout << std::endl;
}

void Matrix::resize(int I, int J) {
    delete_matrix();
    this->I = I;
    this->J = J;
    init_matrix();
}

Matrix* Matrix::Ones(int I) {
    Matrix *R = new Matrix(I, 1);
    for(int i=0 ; i<I ; i++) R->operator()(i, 0) = 1;
    return R;
}

Matrix Matrix::Identity(int I) {
    Matrix R(I, I);
    for(int i=0 ; i<I ; i++) R(i, i) = 1;
    return R;
}

float Matrix::operator()(int i, int j) const {
    return matrix[i*J + j];
}

float& Matrix::operator()(int i, int j) {
    return matrix[i*J + j];
}

Matrix* Matrix::operator*(float lambda) {
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] *= lambda;
        }
    }
    return this;
}

Matrix* Matrix::operator*(const Matrix* B) {
    if(B->I!=J) std::cout << "Matrix dimension dismatch! operator*" << std::endl;
    Matrix *res = new Matrix(I, B->J);
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

Matrix* Matrix::operator+(const Matrix* B) {
    if(B->I!=I || B->J!=J) std::cout << "Matrix dimension dismatch! operator+" << std::endl;
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] += B->operator()(i, j);
        }
    }
    return this;
}

Matrix* Matrix::operator-(const Matrix* B) {
    if(B->I!=I || B->J!=J) std::cout << "Matrix dimension dismatch! operator-" << std::endl;
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] -= B->operator()(i, j);
        }
    }
    return this;
}

Matrix* Matrix::element_wise_product(const Matrix* B) {
    if(B->I!=I || B->J!=J) std::cout << "Matrix dimension dismatch! element_wise_product" << std::endl;
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] *= B->operator()(i, j);
        }
    }
    return this;
}

Matrix* Matrix::transpose() {
    Matrix copy = *this;
    resize(J, I);
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] = copy(j, i);
        }
    }
    return this;
}
