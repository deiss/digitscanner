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
This class defines a matrix and operations that can be applied to it.
*/

#ifndef Matrix_hpp
#define Matrix_hpp

template<typename T>
class Matrix {

    public:
    
        Matrix();
        Matrix(int, int);
        Matrix(const Matrix&);
        Matrix(const Matrix*);
        ~Matrix();
    
        Matrix& operator=(const Matrix& B);
        bool    operator==(const Matrix* B);
    
        int getI() const { return I; }
        int getJ() const { return J; }
    
 static Matrix* ones_ret_n(int);
 static Matrix  ones_ret_c(int);
 static Matrix  identity(int);
    
        T       operator()(int, int) const;
        T&      operator()(int, int);
    
        Matrix& operator*(T);
        Matrix& operator*(const Matrix&);
        Matrix& operator+(const Matrix&);
        Matrix& operator-(const Matrix&);
        
        void    operator*=(T);
        void    operator*=(const Matrix&);
        void    operator+=(const Matrix&);
        void    operator-=(const Matrix&);
    
        Matrix* operator_times(T);
        Matrix* operator_times(const Matrix*);
        Matrix* operator_plus(const Matrix*);
        Matrix* operator_minus(const Matrix*);
    
        void    delete_matrix();
        Matrix* element_wise_product(const Matrix*);
        Matrix& element_wise_product(const Matrix&);
        void    init_matrix();
        void    print() const;
        void    resize(int, int);
 inline T       sigmoid(T) const;
        Matrix* sigmoid();
        Matrix* transpose_ret_p();
        Matrix& transpose_ret_r();
        Matrix  transpose_ret_c() const;
    
    private:
    
        void transpose();

        int I;        /* number of rows */
        int J;        /* number of columns */
        T*  matrix;   /* matrix' coefficients */

};


/*
Default constructor.
*/
template<typename T>
Matrix<T>::Matrix() {
}

/*
Initializes the variables.
*/
template<typename T>
Matrix<T>::Matrix(int I, int J) :
    I(I), J(J) {
    init_matrix();
}

/*
Initializes this matrix doing a copy of matrix B.
*/
template<typename T>
Matrix<T>::Matrix(const Matrix<T>& B) :
    I(B.I), J(B.J) {
    init_matrix();
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] = B(i, j);
        }
    }
}

/*
Initializes this matrix doing a copy of matrix B.
*/
template<typename T>
Matrix<T>::Matrix(const Matrix<T>* B) :
    I(B->I), J(B->J) {
    init_matrix();
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] = B->operator()(i, j);
        }
    }
}

/*
This matrix's coefficient are the same as B's (same pointer).
*/
template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& B) {
    I      = B.I;
    J      = B.J;
    matrix = B.matrix;
}

/*
Comparison operator.
*/
template<typename T>
bool Matrix<T>::operator==(const Matrix<T>* B) {
    if(I!=B->getI() || J!=B->getJ()) {
        return false;
    }
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            if(matrix[i*J + j]!=B->operator()(i, j)) {
                return false;
            }
        }
    }
    return true;
}

/*
Deletes the coefficients.
*/
template<typename T>
Matrix<T>::~Matrix() {
    delete_matrix();
}

/*
Applies the sigmoid function to a number.
*/
template<typename T>
T Matrix<T>::sigmoid(T x) const {
    return 1/(1+exp(-x));
}

/*
Applies the sigmoid function to a matrix.
*/
template<typename T>
Matrix<T>* Matrix<T>::sigmoid() {
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] = sigmoid(matrix[i*J + j]);
        }
    }
    return this;
}

/*
Deletes the coefficients.
*/
template<typename T>
void Matrix<T>::delete_matrix() {
    delete [] matrix;
}

/*
Sets the matrix' coefficients to 0.
*/
template<typename T>
void Matrix<T>::init_matrix() {
    matrix = new T[I*J];
    for(int i=0 ; i<I*J ; i++) {
        matrix[i] = 0;
    }
}

/*
Displays the matrix.
*/
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

/*
Resizes the matrix. The old martrix is deleted and the new one
has all coefficients to 0.
*/
template<typename T>
void Matrix<T>::resize(int I, int J) {
    delete_matrix();
    this->I = I;
    this->J = J;
    init_matrix();
}

/*
Dynamically creates a vector of ones and return a pointer.
*/
template<typename T>
Matrix<T>* Matrix<T>::ones_ret_n(int I) {
    Matrix* R = new Matrix(I, 1);
    for(int i=0 ; i<I ; i++) R->operator()(i, 0) = 1;
    return R;
}
/*
Dynamically creates a vector of ones and returns a copy.
*/
template<typename T>
Matrix<T> Matrix<T>::ones_ret_c(int I) {
    Matrix R(I, 1);
    for(int i=0 ; i<I ; i++) R(i, 0) = 1;
    return R;
}

/*
Creates the identity matrix.
*/
template<typename T>
Matrix<T> Matrix<T>::identity(int I) {
    Matrix R(I, I);
    for(int i=0 ; i<I ; i++) R(i, i) = 1;
    return R;
}

/*
Returns the matrix coefficient at row i, column j.
*/
template<typename T>
T Matrix<T>::operator()(int i, int j) const {
    return matrix[i*J + j];
}

/*
Returns a reference to the matrix coefficient at row i, column j.
*/
template<typename T>
T& Matrix<T>::operator()(int i, int j) {
    return matrix[i*J + j];
}

/* Reference operators */

/*
Multiplication of a matrix by a number.
*/
template<typename T>
Matrix<T>& Matrix<T>::operator*(T lambda) {
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] *= lambda;
        }
    }
    return *this;
}
template<typename T>
void Matrix<T>::operator*=(T lambda) {
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] *= lambda;
        }
    }
}
template<typename T>
Matrix<T>* Matrix<T>::operator_times(T lambda) {
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] *= lambda;
        }
    }
    return this;
}

/*
Product of two matrices, can only be called on dynamically created
(using 'new') Matrix objects.
*/
template<typename T>
Matrix<T>& Matrix<T>::operator*(const Matrix& B) {
    if(B.I!=J) std::cerr << "Matrix dimension dismatch! operator*" << std::endl;
    Matrix res(I, B.J);
    for(int i=0 ; i<I ; i++) {
        for(int k=0 ; k<B.I ; k++) {
            for(int j=0 ; j<B.J ; j++) {
                res(i, j) += matrix[i*J + k]*B(k, j);
            }
        }
    }
    *this = res;
    return *this;
}
template<typename T>
void Matrix<T>::operator*=(const Matrix& B) {
    if(B.I!=J) std::cerr << "Matrix dimension dismatch! operator*" << std::endl;
    Matrix res(I, B.J);
    for(int i=0 ; i<I ; i++) {
        for(int k=0 ; k<B.I ; k++) {
            for(int j=0 ; j<B.J ; j++) {
                res(i, j) += matrix[i*J + k]*B(k, j);
            }
        }
    }
    *this = res;
}
template<typename T>
Matrix<T>* Matrix<T>::operator_times(const Matrix* B) {
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

/*
Addition of two matrices.
*/
template<typename T>
Matrix<T>& Matrix<T>::operator+(const Matrix& B) {
    if(B.I!=I || B.J!=J) std::cerr << "Matrix dimension dismatch! operator+" << std::endl;
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] += B(i, j);
        }
    }
    return *this;
}
template<typename T>
void Matrix<T>::operator+=(const Matrix& B) {
    if(B.I!=I || B.J!=J) std::cerr << "Matrix dimension dismatch! operator+" << std::endl;
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] += B(i, j);
        }
    }
}
template<typename T>
Matrix<T>* Matrix<T>::operator_plus(const Matrix* B) {
    if(B->I!=I || B->J!=J) std::cerr << "Matrix dimension dismatch! operator+" << std::endl;
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] += B->operator()(i, j);
        }
    }
    return this;
}

/*
Substraction of two matrices.
*/
template<typename T>
Matrix<T>& Matrix<T>::operator-(const Matrix& B) {
    if(B.I!=I || B.J!=J) std::cerr << "Matrix dimension dismatch! operator-" << std::endl;
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] -= B(i, j);
        }
    }
    return *this;
}
template<typename T>
void Matrix<T>::operator-=(const Matrix& B) {
    if(B.I!=I || B.J!=J) std::cerr << "Matrix dimension dismatch! operator-" << std::endl;
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] -= B(i, j);
        }
    }
    return *this;
}
template<typename T>
Matrix<T>* Matrix<T>::operator_minus(const Matrix* B) {
    if(B->I!=I || B->J!=J) std::cerr << "Matrix dimension dismatch! operator-" << std::endl;
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] -= B->operator()(i, j);
        }
    }
    return this;
}

/*
Element wise product of two matrices.
*/
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
Matrix<T>& Matrix<T>::element_wise_product(const Matrix& B) {
    if(B.I!=I || B.J!=J) std::cerr << "Matrix dimension dismatch! element_wise_product" << std::endl;
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] *= B(i, j);
        }
    }
    return *this;
}

/*
Transpose the matrix.
*/
template<typename T>
Matrix<T>* Matrix<T>::transpose_ret_p() {
    transpose();
    return this;
}
template<typename T>
Matrix<T>& Matrix<T>::transpose_ret_r() {
    transpose();
    return *this;
}
template<typename T>
Matrix<T> Matrix<T>::transpose_ret_c() const {
    Matrix copy(this);
    copy.transpose();
    return copy;
}
template<typename T>
void Matrix<T>::transpose() {
    Matrix copy = *this;
    resize(J, I);
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] = copy(j, i);
        }
    }
}

#endif
