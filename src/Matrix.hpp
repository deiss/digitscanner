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
Matrices defined with this class are pointers. They all point to an
array of coefficients. One array of coefficients in memory can be
common to several matrices.

This was intented for the following reasons. Doing so avoids allocating
and freeing memory when objects are copied. This could, of course, have
been achieved using pointer to Matrix object: however, it would have
therefore been impossible to use the operator overriding in a nice way.
With this technique, the following instructions only allocate memory
for two matrices, M1 and M2. M1_copy and M2_copy matrices just point to
the same array of coefficients as M1 and M2:

    Matrix<double> M1(3, 4);   M1.fill(1);
    Matrix<double> M2(3, 4);   M2.fill(2);
    Matrix<double> M1_copy = M1;
    Matrix<double> M2_copy = M2;
    M2_copie *= 2.5;
    M2_copie += M1;
    M2_copie.transpose();
    
To get the same behavior without this feature, it would have been necessary
to use the complete call to the operators (M1->operator*=(M2)), which makes
the code hard to read.

A few rules need to be taken when using this class. They are listed below.

Creating a new matrix:
    If you want to create a new matrix, which is a copy of another matrix,
    but you want the two matrices to be seperated in memory, you cannot call
        Matrix<double> M2 = M1;    // M1 points to M2's array
        Matrix<double> M2(M1);     // M1 points to M2's array
    Instead, use the following:
        Matrix<double> M2(M1, true);
    This creates a "deep copy" of M2.

Using the simple operators + - *:
    When using these operators, a new matrix is created in memory and returned.
    This can be inneficient if called many times. Instead, if possible, prefer
    the +=, -= and *=, operators that use the existing matrices in memory to
    do the computation.
 
Memory freeing:
    When not used anymore, you need to manually delete the matrix' coefficients.
    You can do so by calling free() on that matrix. Be careful though that a
    call to free will delete the coefficients of all the matrices that were
    poiting to these coefficients in memory.
    
Function names:
    Functions element_wise_product, sigmoid, transpose, and functions whose
    name begin with 'self' are computed on the matrix. No additional memory is
    allocated. Functions whose name begin with 'create' dupplicate the matrix
    before performing the computation, then return this matrix. They do not
    modify the original matrix but consume more memory.
    
Matrix initialization:
    When creating a matrix, if this matrix is not a copy of another one, memory
    is allocated for the array of coefficients, but they aren't set to 0 by
    default, in case this is useless. This saves computing time. If you want to
    initialize a matrix, you can call the fill or identity functions.
    
Exceptions:
    When an operation is asked on incompatible matrices, an exception of type
    Matrix::Exception is launched. It contains a description of the exception,
    the name of the function where this happened, and informations on the
    matrices involved in the exception.
*/

#ifndef Matrix_hpp
#define Matrix_hpp

#include <exception>
#include <sstream>

template<typename T>
class Matrix {

    public:
    
        Matrix();
        Matrix(int, int);
        Matrix(const Matrix&, bool=false);
        Matrix(const Matrix*, bool=false);
        ~Matrix();
    
        void    set_dimensions(int, int);
    
        Matrix& operator=(const Matrix& B);
    
        bool    operator==(const Matrix& B) const;
        bool    operator==(const Matrix* B) const;
    
        int     get_I() const { if(transpose) return J; else return I; }
        int     get_J() const { if(transpose) return I; else return J; }
    
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
    
        void    element_wise_product(const Matrix*);
        void    element_wise_product(const Matrix&);
        void    sigmoid();
    
        void    self_transpose();
        Matrix  create_transpose() const;
    
        void    fill(T);
        void    identity();
    
 inline T       sigmoid(T) const;
 
        void    free();
        void    print() const;
    
    private:
    
        void copy_matrix(const Matrix<T>*);
        void create_matrix();

        int  I;           /* number of rows */
        int  J;           /* number of columns */
        T*   matrix;      /* matrix' coefficients */
        bool transpose;   /* tells whether the matrix is transposed or not */
    
    
    
    /*
    This class defines an exception that can be thrown when performing an operation
    on matrix with incompatible sizes.
    */
    
    public:
    
        class Exception: public std::exception {

            public:

                Exception(std::string const& p_description, std::string const& p_function, std::string const& p_infos) throw() :
                    description(p_description),
                    function(p_function),
                    infos(p_infos) {
                }
        virtual ~Exception() throw() {}
            
         static std::string create_infos_one_matrix(const Matrix<T>* A) {
                    std::stringstream s_A;   s_A   << (void*)A;         std::string str_A(s_A.str());
                    std::stringstream s_A_m; s_A_m << (void*)A->matrix; std::string str_A_m(s_A_m.str());
                    return "matrix: ["      + str_A + ", " +
                               "I:"         + std::to_string(A->I) + ", " +
                               "J:"         + std::to_string(A->J) + ", " +
                               "matrix:"    + str_A_m +
                               "transpose:" + std::to_string(A->transpose) +
                           "]";
                }
         static std::string create_infos_two_matrices(const Matrix<T>* A, const Matrix<T>* B) {
                    std::stringstream s_A;   s_A   << (void*)A;         std::string str_A(s_A.str());
                    std::stringstream s_B;   s_B   << (void*)B;         std::string str_B(s_B.str());
                    std::stringstream s_A_m; s_A_m << (void*)A->matrix; std::string str_A_m(s_A_m.str());
                    std::stringstream s_B_m; s_B_m << (void*)B->matrix; std::string str_B_m(s_B_m.str());
                    return "A: [" + str_A + ", " +
                                   "I:"         + std::to_string(A->I) + ", " +
                                   "J:"         + std::to_string(A->J) + ", " +
                                   "matrix:"    + str_A_m +
                                   "transpose:" + std::to_string(A->transpose) +
                                "] " +
                           "B: ["               + str_B + ", " +
                                   "I:"         + std::to_string(B->I) + ", " +
                                   "J:"         + std::to_string(B->J) + ", " +
                                   "matrix:"    + str_B_m +
                                   "transpose:" + std::to_string(B->transpose) +
                                "]";
                }
            
        virtual const char* what()      const throw() { return description.c_str(); }
                std::string get_infos() const throw() { return infos; }

            private:

                std::string description;   /* the description of the error */
                std::string function;      /* the declaration of the function where it happened */
                std::string infos;         /* informations about the matrices involved in the exception */

        };

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
Initializes the variables and creates the matrix of coefficients.
*/
template<typename T>
Matrix<T>::Matrix(int I, int J) :
    I(I),
    J(J),
    matrix{0},
    transpose(false) {
    create_matrix();
}

/*
Initializes this matrix by doing a copy of matrix B.
Deep copy means that the matrix of coefficients is dupplicated.
If set to false, this new matrix acts like a pointer that points
to the same matrix of coefficients of matrix B.
*/
template<typename T>
Matrix<T>::Matrix(const Matrix<T>& B, bool deep_copy) :
    I(B.I),
    J(B.J),
    matrix{0} {
    if(deep_copy) { create_matrix(); copy_matrix(&B); transpose = B.transpose; }
    else          { matrix=B.matrix; transpose = B.transpose; }
}
template<typename T>
Matrix<T>::Matrix(const Matrix<T>* B, bool deep_copy) :
    I(B->I),
    J(B->J),
    matrix{0} {
    if(deep_copy) { create_matrix(); copy_matrix(B); transpose = B->transpose; }
    else          { matrix=B->matrix; transpose = B->transpose; }
}

/*
Allows to set the size. Should be used if a matrix is created with
no size parameters. This is not a resizing method!
*/
template<typename T>
void Matrix<T>::set_dimensions(int p_I, int p_J) {
    I = p_I;
    J = p_J;
    free();
    create_matrix();
}

/*
Copies the coefficients matrix.
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
The matrix of coefficients of this matrix are the same as B's.
This matrix points at this array in memory.
*/
template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& B) {
    I         = B.get_I();
    J         = B.get_J();
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
        if(J!=B.get_I() || I!=B.get_J()) {
            return false;
        }
        int BI = B.get_I();
        int BJ = B.get_J();
        if(!B.transpose) {
            for(int i=0 ; i<J ; i++) {
                for(int j=0 ; j<I ; j++) {
                    if(matrix[j*J + i]!=B.matrix[i*BJ + j]) {
                        return false;
                    }
                }
            }
        }
        else {
            for(int i=0 ; i<J ; i++) {
                for(int j=0 ; j<I ; j++) {
                    if(matrix[j*J + i]!=B.matrix[j*BI + i]) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
    else {
        if(I!=B.get_I() || J!=B.get_J()) {
            return false;
        }
        int BI = B.get_I();
        int BJ = B.get_J();
        if(!B.transpose) {
            for(int i=0 ; i<I ; i++) {
                for(int j=0 ; j<J ; j++) {
                    if(matrix[i*J + j]!=B.matrix[i*BJ + j]) {
                        return false;
                    }
                }
            }
        }
        else {
            for(int i=0 ; i<I ; i++) {
                for(int j=0 ; j<J ; j++) {
                    if(matrix[i*J + j]!=B.matrix[j*BI + i]) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
}
template<typename T>
bool Matrix<T>::operator==(const Matrix<T>* B) const {
    if(this==B) {
        return true;
    }
    return *this==*B;
}

/*
Deletes the matrix of coefficients.
*/
template<typename T>
void Matrix<T>::free() {
    if(matrix) {
        delete [] matrix;
        matrix = 0;
    }
}

/*
Default destructor.
*/
template<typename T>
Matrix<T>::~Matrix() {
}

/*
Returns the sigmoid function of a number.
*/
template<typename T>
T Matrix<T>::sigmoid(T x) const {
    return 1/(1+exp(-x));
}

/*
Applies the sigmoid function to a matrix, element-wise.
*/
template<typename T>
void Matrix<T>::sigmoid() {
    for(int i=0 ; i<I ; i++) {
        for(int j=0 ; j<J ; j++) {
            matrix[i*J + j] = sigmoid(matrix[i*J + j]);
        }
    }
}

/*
Allocates memory for the matrix of coefficients.
*/
template<typename T>
void Matrix<T>::create_matrix() {
    try {
        matrix = new T[I*J];
    }
    catch(std::exception exc) {
        std::string description = "Unable to allocate memory for the matrix: " + std::string(exc.what());
        std::string function    = "void Matrix<T>::create_matrix()";
        std::string infos       = Exception::create_infos_one_matrix(this);
        Exception   e(description, function, infos);
        throw e;
    }
}

/*
Fills the matrix with a scalar.
*/
template<typename T>
void Matrix<T>::fill(T alpha) {
    for(int i=0 ; i<I*J ; i++) {
        matrix[i] = alpha;
    }
}

/*
Creates the identity matrix.
This function throws and exception if the matrix is not
a square matrix.
*/
template<typename T>
void Matrix<T>::identity() {
    fill(0);
    if(I!=J) {
        std::string description = "Unable to create identity matrix, this is not a square matrix.";
        std::string function    = "void Matrix<T>::fill_identity()";
        std::string infos       = "matrix: [" << std::to_string(this) << ", I:" << I << ", J:" << J << ", matrix:" << matrix << "transpose:" << transpose << "]";
        Exception   e(description, function, infos);
        throw e;
    }
    for(int i=0 ; i<I ; i++) {
        matrix[i*(I+1)] = 1;
    }
}

/*
Displays the matrix.
*/
template<typename T>
void Matrix<T>::print() const {
    if(!transpose) {
        for(int i=0 ; i<I ; i++) {
            std::cout << "| ";
            for(int j=0 ; j<J ; j++) {
                std::cout << matrix[i*J + j] << " ";
            }
            std::cout << "|" << std::endl;
        }
        std::cout << std::endl;
    }
    else {
        for(int i=0 ; i<J ; i++) {
            std::cout << "| ";
            for(int j=0 ; j<I ; j++) {
                std::cout << matrix[j*J + i] << " ";
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
    if(!transpose) {
        return matrix[i*J + j];
    }
    else {
        return matrix[j*J + i];
    }
}

/*
Returns a reference to the matrix coefficient at row i, column j.
*/
template<typename T>
T& Matrix<T>::operator()(int i, int j) {
    if(!transpose) {
        return matrix[i*J + j];
    }
    else {
        return matrix[j*J + i];
    }
}

/*
Multiplication of a matrix by a scalar.
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
    if(!transpose) {
        if(B.get_I()!=J) {
            std::string desc     = "Unable to multiply these two matrices (A*B): dimensions don't match.";
            std::string function = "void Matrix<T>::operator*=(const Matrix& B)";
            std::string infos    = Exception::create_infos_two_matrices(this, &B);
            Exception   e(desc, function, infos);
            throw e;
        }
        Matrix res(I, B.get_J());
        res.fill(0);
        int BI = B.get_I();
        int BJ = B.get_J();
        if(!B.transpose) {
            for(int i=0 ; i<I ; i++) {
                for(int k=0 ; k<BI ; k++) {
                    for(int j=0 ; j<BJ ; j++) {
                        res.matrix[i*BJ + j] += matrix[i*J + k]*B.matrix[k*BJ + j];
                    }
                }
            }
        }
        else {
            for(int i=0 ; i<I ; i++) {
                for(int k=0 ; k<BI ; k++) {
                    for(int j=0 ; j<BJ; j++) {
                        res.matrix[i*BJ + j] += matrix[i*J + k]*B.matrix[j*BI + k];
                    }
                }
            }
        }
        free();
        *this = res;
    }
    else {
        if(B.get_I()!=I) {
            std::string desc     = "Unable to multiply these two matrices (A*B): dimensions don't match.";
            std::string function = "void Matrix<T>::operator*=(const Matrix& B)";
            std::string infos    = Exception::create_infos_two_matrices(this, &B);
            Exception    e(desc, function, infos);
            throw e;
        }
        Matrix res(J, B.get_J());
        res.fill(0);
        int BI = B.get_I();
        int BJ = B.get_J();
        if(!B.transpose) {
            for(int i=0 ; i<J ; i++) {
                for(int k=0 ; k<BI ; k++) {
                    for(int j=0 ; j<BJ ; j++) {
                        res.matrix[i*BJ + j] += matrix[k*J + i]*B.matrix[k*BJ + j];
                    }
                }
            }
        }
        else {
            for(int i=0 ; i<J ; i++) {
                for(int k=0 ; k<BI ; k++) {
                    for(int j=0 ; j<BJ ; j++) {
                        res.matrix[i*BJ + j] += matrix[k*J + i]*B.matrix[j*BI + k];
                    }
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
    if(!transpose) {
        if(B.get_I()!=J) {
            std::string desc     = "Unable to multiply these two matrices (A*B): dimensions don't match.";
            std::string function = "void Matrix<T>::operator*=(const Matrix& B)";
            std::string infos    = Exception::create_infos_two_matrices(this, &B);
            Exception   e(desc, function, infos);
            throw e;
        }
        Matrix res(I, B.get_J());
        res.fill(0);
        int BI = B.get_I();
        int BJ = B.get_J();
        if(!B.transpose) {
            for(int i=0 ; i<I ; i++) {
                for(int k=0 ; k<BI ; k++) {
                    for(int j=0 ; j<BJ ; j++) {
                        res.matrix[i*BJ + j] += matrix[i*J + k]*B.matrix[k*BJ + j];
                    }
                }
            }
        }
        else {
            for(int i=0 ; i<I ; i++) {
                for(int k=0 ; k<BI ; k++) {
                    for(int j=0 ; j<BJ; j++) {
                        res.matrix[i*BJ + j] += matrix[i*J + k]*B.matrix[j*BI + k];
                    }
                }
            }
        }
        return res;
    }
    else {
        if(B.get_I()!=I) {
            std::string desc     = "Unable to multiply these two matrices (A*B): dimensions don't match.";
            std::string function = "void Matrix<T>::operator*=(const Matrix& B)";
            std::string infos    = Exception::create_infos_two_matrices(this, &B);
            Exception    e(desc, function, infos);
            throw e;
        }
        Matrix res(J, B.get_J());
        res.fill(0);
        int BI = B.get_I();
        int BJ = B.get_J();
        if(!B.transpose) {
            for(int i=0 ; i<J ; i++) {
                for(int k=0 ; k<BI ; k++) {
                    for(int j=0 ; j<BJ ; j++) {
                        res.matrix[i*BJ + j] += matrix[k*J + i]*B.matrix[k*BJ + j];
                    }
                }
            }
        }
        else {
            for(int i=0 ; i<J ; i++) {
                for(int k=0 ; k<BI ; k++) {
                    for(int j=0 ; j<BJ ; j++) {
                        res.matrix[i*BJ + j] += matrix[k*J + i]*B.matrix[j*BI + k];
                    }
                }
            }
        }
        return res;
    }
}
template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix* B) const {
    return (*this)*(*B);
}

/*
Addition of two matrices.
*/
template<typename T>
void Matrix<T>::operator+=(const Matrix& B) {
    if(!transpose) {
        if(B.get_I()!=I || B.get_J()!=J) {
            std::string desc     = "Unable to add these two matrices (A+B): dimensions don't match.";
            std::string function = "void Matrix<T>::operator+=(const Matrix& B)";
            std::string infos    = Exception::create_infos_two_matrices(this, &B);
            Exception   e(desc, function, infos);
            throw e;
        }
        int BI = B.get_I();
        int BJ = B.get_J();
        if(!B.transpose) {
            for(int i=0 ; i<I ; i++) {
                for(int j=0 ; j<J ; j++) {
                    matrix[i*J + j] += B.matrix[i*BJ + j];
                }
            }
        }
        else {
            for(int i=0 ; i<I ; i++) {
                for(int j=0 ; j<J ; j++) {
                    matrix[i*J + j] += B.matrix[j*BI + i];
                }
            }
        }
    }
    else {
        if(B.get_I()!=J || B.get_J()!=I) {
            std::string desc     = "Unable to add these two matrices (A+B): dimensions don't match.";
            std::string function = "void Matrix<T>::operator+=(const Matrix& B)";
            std::string infos    = Exception::create_infos_two_matrices(this, &B);
            Exception   e(desc, function, infos);
            throw e;
        }
        int BI = B.get_I();
        int BJ = B.get_J();
        if(!B.transpose) {
            for(int i=0 ; i<J ; i++) {
                for(int j=0 ; j<I ; j++) {
                    matrix[j*J + i] += B.matrix[i*BJ + j];
                }
            }
        }
        else {
            for(int i=0 ; i<J ; i++) {
                for(int j=0 ; j<I ; j++) {
                    matrix[j*J + i] += B.matrix[j*BI + i];
                }
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
    if(!transpose) {
        if(B.get_I()!=I || B.get_J()!=J) {
            std::string desc     = "Unable to substract these two matrices (A-B): dimensions don't match.";
            std::string function = "void Matrix<T>::operator-=(const Matrix& B)";
            std::string infos    = Exception::create_infos_two_matrices(this, &B);
            Exception   e(desc, function, infos);
            throw e;
        }
        int BI = B.get_I();
        int BJ = B.get_J();
        if(!B.transpose) {
            for(int i=0 ; i<I ; i++) {
                for(int j=0 ; j<J ; j++) {
                    matrix[i*J + j] -= B.matrix[i*BJ + j];
                }
            }
        }
        else {
            for(int i=0 ; i<I ; i++) {
                for(int j=0 ; j<J ; j++) {
                    matrix[i*J + j] -= B.matrix[j*BI + i];
                }
            }
        }
    }
    else {
        if(B.get_I()!=J || B.get_J()!=I) {
            std::string desc     = "Unable to substract these two matrices (A-B): dimensions don't match.";
            std::string function = "void Matrix<T>::operator-=(const Matrix& B)";
            std::string infos    = Exception::create_infos_two_matrices(this, &B);
            Exception   e(desc, function, infos);
            throw e;
        }
        int BI = B.get_I();
        int BJ = B.get_J();
        if(!B.transpose) {
            for(int i=0 ; i<J ; i++) {
                for(int j=0 ; j<I ; j++) {
                    matrix[j*J + i] -= B.matrix[i*BJ + j];
                }
            }
        }
        else {
            for(int i=0 ; i<J ; i++) {
                for(int j=0 ; j<I ; j++) {
                    matrix[j*J + i] -= B.matrix[j*BI + i];
                }
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
Element wise product of two matrices (Hadamard product).
*/
template<typename T>
void Matrix<T>::element_wise_product(const Matrix* B) {
    element_wise_product(*B);
}
template<typename T>
void Matrix<T>::element_wise_product(const Matrix& B) {
    if(!transpose) {
        if(B.get_I()!=I || B.get_J()!=J) {
            std::string desc     = "Unable to perform Hadamard product with these two matrices (A°B): dimensions don't match.";
            std::string function = "void Matrix<T>::self_element_wise_product(const Matrix& B)";
            std::string infos    = Exception::create_infos_two_matrices(this, &B);
            Exception   e(desc, function, infos);
            throw e;
        }
        int BI = B.get_I();
        int BJ = B.get_J();
        if(!B.transpose) {
            for(int i=0 ; i<I ; i++) {
                for(int j=0 ; j<J ; j++) {
                    matrix[i*J + j] *= B.matrix[i*BJ + j];
                }
            }
        }
        else {
            for(int i=0 ; i<I ; i++) {
                for(int j=0 ; j<J ; j++) {
                    matrix[i*J + j] *= B.matrix[j*BI + i];
                }
            }
        }
    }
    else {
        if(B.get_I()!=J || B.get_J()!=I) {
            std::string desc     = "Unable to perform Hadamard product with these two matrices (A°B): dimensions don't match.";
            std::string function = "void Matrix<T>::self_element_wise_product(const Matrix& B)";
            std::string infos    = Exception::create_infos_two_matrices(this, &B);
            Exception   e(desc, function, infos);
            throw e;
        }
        int BI = B.get_I();
        int BJ = B.get_J();
        if(!B.transpose) {
            for(int i=0 ; i<J ; i++) {
                for(int j=0 ; j<I ; j++) {
                    matrix[j*J + i] *= B.matrix[i*BJ + j];
                }
            }
        }
        else {
            for(int i=0 ; i<J ; i++) {
                for(int j=0 ; j<I ; j++) {
                    matrix[j*J + i] *= B.matrix[j*BI + i];
                }
            }
        }
    }
}

/*
Creates a new matrix which is the transposed of this one and returns it.
*/
template<typename T>
Matrix<T> Matrix<T>::create_transpose() const {
    Matrix Mt(this, true);
    if(!Mt.transpose) Mt.transpose = true;
    else              Mt.transpose = false;
    return Mt;
}

/*
Transposes this matrix.
*/
template<typename T>
void Matrix<T>::self_transpose() {
    if(!transpose) transpose = true;
    else           transpose = false;
}

#endif
