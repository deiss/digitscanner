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
    
        bool    operator==(const Matrix& B) const;
        bool    operator==(const Matrix* B) const;
    
        int     getI() const { if(transpose) return J; else return I; }
        int     getJ() const { if(transpose) return I; else return J; }
    
        T       operator()(int, int)  const;
        T&      operator()(int, int);
    
        void    operator*=(T);
        Matrix  operator*(T)   const;
    
        void    operator*=(const Matrix&);
        void    operator*=(const Matrix*);
        Matrix  operator*(const Matrix&)   const;
        Matrix  operator*(const Matrix*)   const;
    
        void    operator+=(const Matrix&);
        void    operator+=(const Matrix*);
        Matrix  operator+(const Matrix&)   const;
        Matrix  operator+(const Matrix*)   const;
    
        void    operator-=(const Matrix&);
        void    operator-=(const Matrix*);
        Matrix  operator-(const Matrix&)   const;
        Matrix  operator-(const Matrix*)   const;
    
        Matrix  create_transpose() const;
    
        void    self_element_wise_product(const Matrix*);
        void    self_element_wise_product(const Matrix&);
        void    self_sigmoid();
        void    self_transpose();
    
        void    fill(T);
        void    fill_identity();
    
 inline T       sigmoid(T) const;
 
        void    free();
        void    print() const;
    
    private:
    
        void copy_matrix(const Matrix<T>*);
        void init_matrix();

        int  I;           /* number of rows */
        int  J;           /* number of columns */
        T*   matrix;      /* matrix' coefficients */
        bool transpose;   /* tells whether the matrix is transposed or not */

};



/*
Default constructor.
*/
template<typename T>
Matrix<T>::Matrix() :
    I(0),
    J(0),
    matrix{0},
    transpose(false) {
}

/*
Initializes the variables.
*/
template<typename T>
Matrix<T>::Matrix(int I, int J) :
    I(I),
    J(J),
    matrix{0},
    transpose(false) {
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
    if(deep_copy) { init_matrix(); copy_matrix(&B); transpose = B.transpose; }
    else          { matrix=B.matrix; transpose = B.transpose; }
}
template<typename T>
Matrix<T>::Matrix(const Matrix<T>* B, bool deep_copy) :
    I(B->I),
    J(B->J),
    matrix{0} {
    if(deep_copy) { init_matrix(); copy_matrix(B); transpose = B->transpose; }
    else          { matrix=B->matrix; transpose = B->transpose; }
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
    I         = B.getI();
    J         = B.getJ();
    matrix    = B.matrix;
    transpose = B.transpose;
    return *this;
}

/*
Comparison operator.
*/
template<typename T>
bool Matrix<T>::operator==(const Matrix<T>& B) const {
    if(transpose) {
        if(J!=B.getI() || I!=B.getJ()) {
            return false;
        }
        for(int i=0 ; i<J ; i++) {
            for(int j=0 ; j<I ; j++) {
                if(matrix[j*J + i]!=B(i, j)) {
                    return false;
                }
            }
        }
        return true;
    }
    else {
        if(I!=B.getI() || J!=B.getJ()) {
            return false;
        }
        for(int i=0 ; i<I ; i++) {
            for(int j=0 ; j<J ; j++) {
                if(matrix[i*J + j]!=B(i, j)) {
                    return false;
                }
            }
        }
        return true;
    }
}
template<typename T>
bool Matrix<T>::operator==(const Matrix<T>* B) const {
    return *this==*B;
}

/*
Deletes the coefficients.
*/
template<typename T>
void Matrix<T>::free() {
    if(matrix) {
        delete [] matrix;
        matrix = 0;
    }
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
Applies the sigmoid function to a matrix. This function is the same
whether the matrix is transposed or not.
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
Creates the coefficients matrix.
*/
template<typename T>
void Matrix<T>::init_matrix() {
    matrix = new T[I*J];
}

/*
Fills with zeros.
*/
template<typename T>
void Matrix<T>::fill(T alpha) {
    for(int i=0 ; i<I*J ; i++) {
        matrix[i] = alpha;
    }
}

/*
Creates the identity matrix.
*/
template<typename T>
void Matrix<T>::fill_identity() {
    if(I!=J) std::cerr << "fill_identity(): Not a squared matrix" << std::endl;
    for(int i=0 ; i<I ; i++) {
        matrix[i*(I+1)] = 1;
    }
}

/*
Displays the matrix.
*/
template<typename T>
void Matrix<T>::print() const {
    if(transpose) {
        for(int i=0 ; i<J ; i++) {
            std::cout << "| ";
            for(int j=0 ; j<I ; j++) {
                std::cout << matrix[j*J + i] << " ";
            }
            std::cout << "|" << std::endl;
        }
        std::cout << std::endl;
    }
    else {
        for(int i=0 ; i<I ; i++) {
            std::cout << "| ";
            for(int j=0 ; j<J ; j++) {
                std::cout << matrix[i*J + j] << " ";
            }
            std::cout << "|" << std::endl;
        }
        std::cout << std::endl;
    }
}

/*
Returns the matrix coefficient at row i, column j.
*/
template<typename T>
T Matrix<T>::operator()(int i, int j) const {
    if(transpose) {
        return matrix[j*J + i];
    }
    else {
        return matrix[i*J + j];
    }
}

/*
Returns a reference to the matrix coefficient at row i, column j.
*/
template<typename T>
T& Matrix<T>::operator()(int i, int j) {
    if(transpose) {
        return matrix[j*J + i];
    }
    else {
        return matrix[i*J + j];
    }
}

/*
Multiplication of a matrix by a number.
*/
template<typename T>
Matrix<T> Matrix<T>::operator*(T lambda) const {
    Matrix<T> result(this, true);
    result *= lambda;
    return result;
}
template<typename T>
void Matrix<T>::operator*=(T lambda) {
    for(int i=0 ; i<I*J ; i++) {
        matrix[i] *= lambda;
    }
}

/*
Product of two matrices.
*/
template<typename T>
void Matrix<T>::operator*=(const Matrix& B) {
    if(transpose) {
        if(B.getI()!=I) {
            std::cerr << "Matrix dimension dismatch! operator*" << std::endl;
            throw 1;
        }
        Matrix res(J, B.getJ());
        res.fill(0);
        for(int i=0 ; i<J ; i++) {
            for(int k=0 ; k<B.getI() ; k++) {
                for(int j=0 ; j<B.getJ() ; j++) {
                    res(i, j) += matrix[k*J + i]*B(k, j);
                }
            }
        }
        free();
        *this = res;
    }
    else {
        if(B.getI()!=J) {
            std::cerr << "Matrix dimension dismatch! operator*" << std::endl;
            throw 1;
        }
        Matrix res(I, B.getJ());
        res.fill(0);
        for(int i=0 ; i<I ; i++) {
            for(int k=0 ; k<B.getI() ; k++) {
                for(int j=0 ; j<B.getJ() ; j++) {
                    res(i, j) += matrix[i*J + k]*B(k, j);
                }
            }
        }
        free();
        *this = res;
    }
}
template<typename T>
void Matrix<T>::operator*=(const Matrix* B) {
    *this *= *B;
}
template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix& B) const {
    Matrix result(this, true);
    result *= B;
    return result;
}
template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix* B) const {
    Matrix result(this, true);
    result *= *B;
    return result;
}

/*
Addition of two matrices.
*/
template<typename T>
void Matrix<T>::operator+=(const Matrix& B) {
    if(transpose) {
        if(B.getI()!=J || B.getJ()!=I) {
            std::cerr << "Matrix dimension dismatch! operator+" << std::endl;
            throw 2;
        }
        for(int i=0 ; i<J ; i++) {
            for(int j=0 ; j<I ; j++) {
                matrix[j*J + i] += B(i, j);
            }
        }
    }
    else {
        if(B.getI()!=I || B.getJ()!=J) {
            std::cerr << "Matrix dimension dismatch! operator+" << std::endl;
            throw 2;
        }
        for(int i=0 ; i<I ; i++) {
            for(int j=0 ; j<J ; j++) {
                matrix[i*J + j] += B(i, j);
            }
        }
    }
}
template<typename T>
void Matrix<T>::operator+=(const Matrix* B) {
    *this += *B;
}
template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix& B) const {
    Matrix result(this, true);
    result += B;
    return result;
}
template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix* B) const {
    Matrix result(this, true);
    result += *B;
    return result;
}

/*
Substraction of two matrices.
*/
template<typename T>
void Matrix<T>::operator-=(const Matrix& B) {
    if(transpose) {
        if(B.getI()!=J || B.getJ()!=I) {
            std::cerr << "Matrix dimension dismatch! operator-" << std::endl;
            throw 2;
        }
        for(int i=0 ; i<J ; i++) {
            for(int j=0 ; j<I ; j++) {
                matrix[j*J + i] -= B(i, j);
            }
        }
    }
    else {
        if(B.getI()!=I || B.getJ()!=J) {
            std::cerr << "Matrix dimension dismatch! operator-" << std::endl;
            throw 2;
        }
        for(int i=0 ; i<I ; i++) {
            for(int j=0 ; j<J ; j++) {
                matrix[i*J + j] -= B(i, j);
            }
        }
    }
}
template<typename T>
void Matrix<T>::operator-=(const Matrix* B) {
    *this -= *B;
}
template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix& B) const {
    Matrix<T> result(this, true);
    result -= B;
    return result;
}
template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix* B) const {
    Matrix<T> result(this, true);
    result -= *B;
    return result;
}

/*
Element wise product of two matrices.
*/
template<typename T>
void Matrix<T>::self_element_wise_product(const Matrix* B) {
    self_element_wise_product(*B);
}
template<typename T>
void Matrix<T>::self_element_wise_product(const Matrix& B) {
    if(transpose) {
        if(B.getI()!=J || B.getJ()!=I) {
            std::cerr << "Matrix dimension dismatch! element_wise_product" << std::endl;
            throw 1;
        }
        for(int i=0 ; i<J ; i++) {
            for(int j=0 ; j<I ; j++) {
                matrix[j*J + i] *= B(i, j);
            }
        }
    }
    else {
        if(B.getI()!=I || B.getJ()!=J) {
            std::cerr << "Matrix dimension dismatch! element_wise_product" << std::endl;
            throw 1;
        }
        for(int i=0 ; i<I ; i++) {
            for(int j=0 ; j<J ; j++) {
                matrix[i*J + j] *= B(i, j);
            }
        }
    }
}

/*
Creates a new matrix which is the transposed of this one and returns it.
*/
template<typename T>
Matrix<T> Matrix<T>::create_transpose() const {
    Matrix Mt(this);
    if(Mt.transpose) Mt.transpose = false;
    else             Mt.transpose = true;
    return Mt;
}

/*
In-place transpose of the matrix.
*/
template<typename T>
void Matrix<T>::self_transpose() {
    if(transpose) transpose = false;
    else          transpose = true;
}

#endif
