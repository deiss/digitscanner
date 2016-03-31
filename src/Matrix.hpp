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
        Matrix(const Matrix&, bool=false);
        Matrix(const Matrix*, bool=false);
        ~Matrix();
    
        Matrix& operator=(const Matrix& B);
        bool    operator==(const Matrix* B);
    
        int  getI() const { return I; }
        int  getJ() const { return J; }
        void setI(int p_I) { I = p_I; }
        void setJ(int p_J) { J = p_J; }
    
 static Matrix* ones_ret_n(int);
 static Matrix  ones_ret_c(int);
 static Matrix  identity(int);
    
        T       operator()(int, int) const;
        T&      operator()(int, int);
    
        Matrix  operator*(T)             const;
        Matrix  operator*(const Matrix&) const;
        Matrix  operator*(const Matrix*) const;
        Matrix  operator+(const Matrix&) const;
        Matrix  operator-(const Matrix&) const;
        Matrix  operator-(const Matrix*) const;
        Matrix  create_transpose()       const;
        
        void    operator*=(T);
        void    operator*=(const Matrix&);
        void    operator*=(const Matrix*);
        void    operator+=(const Matrix&);
        void    operator+=(const Matrix*);
        void    operator-=(const Matrix&);
        void    self_element_wise_product(const Matrix*);
        void    self_element_wise_product(const Matrix&);
        void    self_sigmoid();
        void    self_transpose();
    
 inline T       sigmoid(T) const;
 
        void    free();
        void    print() const;
        void    resize(int, int);
    
    private:
    
        void copy_matrix(const Matrix<T>*);
        void init_matrix();

        int I;        /* number of rows */
        int J;        /* number of columns */
        T*  matrix;   /* matrix' coefficients */

};



/*
Default constructor.
*/
template<typename T>
Matrix<T>::Matrix() :
    I(0),
    J(0),
    matrix{0} {
}

/*
Initializes the variables.
*/
template<typename T>
Matrix<T>::Matrix(int I, int J) :
    I(I),
    J(J),
    matrix{0} {
    init_matrix();
}

/*
Initializes this matrix doing a copy of matrix B.
*/
template<typename T>
Matrix<T>::Matrix(const Matrix<T>& B, bool deep_copy) :
    I(B.I),
    J(B.J),
    matrix{0} {
    if(deep_copy) { init_matrix(); copy_matrix(&B); }
    else          { matrix=B.matrix; }
}
template<typename T>
Matrix<T>::Matrix(const Matrix<T>* B, bool deep_copy) :
    I(B->I),
    J(B->J),
    matrix{0} {
    if(deep_copy) { init_matrix(); copy_matrix(B); }
    else          { matrix=B->matrix; }
}

/*
Copies the matrix array.
*/
template<typename T>
void Matrix<T>::copy_matrix(const Matrix<T>* B) {
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
    return *this;
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
void Matrix<T>::free() {
    if(matrix) { delete [] matrix; matrix = 0; }
}

/*
Deletes the coefficients.
*/
template<typename T>
Matrix<T>::~Matrix() {
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
void Matrix<T>::self_sigmoid() {
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] = sigmoid(matrix[i*J + j]);
        }
    }
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
    free();
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
Matrix<T> Matrix<T>::operator*(T lambda) const {
    Matrix<T> result(this, true);
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            result(i, j) = matrix[i*J + j] * lambda;
        }
    }
    return result;
}
template<typename T>
void Matrix<T>::operator*=(T lambda) {
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] *= lambda;
        }
    }
}

/*
Product of two matrices, can only be called on dynamically created
(using 'new') Matrix objects.
*/
template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix& B) const {
    if(B.I!=J) std::cerr << "Matrix dimension dismatch! operator*" << std::endl;
    Matrix result(I, B.J);
    for(int i=0 ; i<I ; i++) {
        for(int k=0 ; k<B.I ; k++) {
            for(int j=0 ; j<B.J ; j++) {
                result(i, j) += matrix[i*J + k]*B(k, j);
            }
        }
    }
    return result;
}
template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix* B) const {
    if(B->I!=J) std::cerr << "Matrix dimension dismatch! operator*" << std::endl;
    Matrix result(I, B->J);
    for(int i=0 ; i<I ; i++) {
        for(int k=0 ; k<B->I ; k++) {
            for(int j=0 ; j<B->J ; j++) {
                result(i, j) += matrix[i*J + k]*B(k, j);
            }
        }
    }
    return result;
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
    free();
    *this = res;
}
template<typename T>
void Matrix<T>::operator*=(const Matrix* B) {
    if(B->I!=J) std::cerr << "Matrix dimension dismatch! operator*" << std::endl;
    Matrix res(I, B->J);
    for(int i=0 ; i<I ; i++) {
        for(int k=0 ; k<B->I ; k++) {
            for(int j=0 ; j<B->J ; j++) {
                res(i, j) += matrix[i*J + k]*B->operator()(k, j);
            }
        }
    }
    free();
    *this = res;
}

/*
Addition of two matrices.
*/
template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix& B) const {
    Matrix result(this, true);
    if(B.I!=I || B.J!=J) {
        std::cerr << "Matrix dimension dismatch! operator+" << std::endl;
        throw 1;
    }
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            result(i, j) = matrix[i*J + j] + B(i, j);
        }
    }
    return result;
}
template<typename T>
void Matrix<T>::operator+=(const Matrix& B) {
    if(B.I!=I || B.J!=J) {
        std::cerr << "Matrix dimension dismatch! operator+" << std::endl;
        throw 2;
    }
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] += B(i, j);
        }
    }
}
template<typename T>
void Matrix<T>::operator+=(const Matrix* B) {
    if(B->I!=I || B->J!=J) {
        std::cerr << "Matrix dimension dismatch! operator+" << std::endl;
        throw 3;
    }
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] += B->operator()(i, j);
        }
    }
}

/*
Substraction of two matrices.
*/
template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix& B) const {
    Matrix<T> result(this, true);
    if(B.I!=I || B.J!=J) std::cerr << "Matrix dimension dismatch! operator-" << std::endl;
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            result(i, j) = matrix[i*J + j] - B(i, j);
        }
    }
    return result;
}
template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix* B) const {
    Matrix<T> result(this, true);
    if(B->I!=I || B->J!=J) std::cerr << "Matrix dimension dismatch! operator-" << std::endl;
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            result(i, j) = matrix[i*J + j] - B->operator()(i, j);
        }
    }
    return result;
}
template<typename T>
void Matrix<T>::operator-=(const Matrix& B) {
    if(B.I!=I || B.J!=J) std::cerr << "Matrix dimension dismatch! operator-" << std::endl;
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] -= B(i, j);
        }
    }
}

/*
Element wise product of two matrices.
*/
template<typename T>
void Matrix<T>::self_element_wise_product(const Matrix* B) {
    if(B->I!=I || B->J!=J) std::cerr << "Matrix dimension dismatch! element_wise_product" << std::endl;
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] *= B->operator()(i, j);
        }
    }
}
template<typename T>
void Matrix<T>::self_element_wise_product(const Matrix& B) {
    if(B.I!=I || B.J!=J) std::cerr << "Matrix dimension dismatch! element_wise_product" << std::endl;
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] *= B(i, j);
        }
    }
}

/*
Creates a new matrix which is the transposed of this one and returns it.
*/
template<typename T>
Matrix<T> Matrix<T>::create_transpose() const {
    Matrix Mt(this);
    Mt.setI(J);
    Mt.setJ(I);
    return Mt;
}

/*
In-place transpose of the matrix.
*/
template<typename T>
void Matrix<T>::self_transpose() {
    int I_old = I;
    I = J;
    J = I_old;
}

#endif
