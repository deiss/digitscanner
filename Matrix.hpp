#ifndef Matrix_h
#define Matrix_h

// template?
class Matrix {

    public:
    
        Matrix(int, int);
        Matrix(const Matrix &);
        Matrix(const Matrix *);
        ~Matrix();
    
        int getI() const { return I; }
        int getJ() const { return J; }
    
 static Matrix *Ones(int);
 static Matrix *Identity(int);
    
        float   operator()(int, int) const ;
        float  &operator()(int, int);
        Matrix *operator*(float);
        Matrix *operator*(const Matrix *);
        Matrix *operator+(const Matrix *);
        Matrix *operator-(const Matrix *);
    
        void    delete_matrix();
        Matrix *element_wise_product(const Matrix *);
        void    init_matrix();
        void    print() const;
        void    resize(int, int);
 inline float   sigmoid(float) const;
        Matrix *sigmoid();
        Matrix *transpose();
    
    private:

        int     I;
        int     J;
        float **matrix;

};

#endif