
#ifndef NN_UTILS_H
#define NN_UTILS_H

#include <vector>
#include <cassert>


struct matrix;
using comp = std::vector<matrix>;
using vecs = std::vector<std::vector<double>>;
//using vec = std::vector<double>;
struct vec{
    int size = 0;
    vec()=default;
    vec (int k);
    std::vector<double> data;

    friend vec sqrt_eps(vec v, double eps);

    void zero();
    void operator*=(double o);
    void operator-=(const vec& o);
    void operator+=(const vec& o);

    friend vec operator*(double a, vec r);
    friend vec operator/(vec l, const vec& r);
    void pow2();
};

struct matrix{

    int height = 0;
    int width = 0;
    vecs data;
    matrix()=default;
    matrix(int k, int l);
    matrix(vecs data);

    /* matrix addition */
    friend matrix operator+(matrix l, const matrix& r);
    friend matrix operator-(matrix l, const matrix& r);
    friend matrix operator*(matrix l, const matrix& r);
    friend matrix operator/(matrix l, const matrix& r);


    friend matrix operator*(double a, matrix r);

    friend matrix sqrt_eps(matrix m, double eps);

    void operator+=(const matrix& o);
    void operator-=(const matrix& o);
    void operator*=(const matrix& o);


    void operator/=(double);
    void operator*=(double o);

    void add_vec(const vec& v);

    void sum_to(vec& res, int axis);

    void zero();

    matrix& exp();

    void pow2();

    friend void matmul(const vecs& data, int start, int end, const matrix& m2, matrix& res);

    // m1 has (k, n) shape, m2 has (n, l) shape, res has (k, l) shape
    friend void matmul(const matrix& m1, const matrix& m2, matrix& res);

    // m1 has (k, n) shape, m2 has (n, l) shape
    friend matrix matmul(const matrix& m1, const matrix& m2);

    friend void matmul_t1(const matrix& m1, const matrix& m2, matrix& res);
    friend void matmul_t1(const vecs& data, int start, int end, const matrix& m2, matrix& res);

    friend void matmul_t2(const matrix& m1, const matrix& m2, matrix& res);

    bool operator==(const matrix& o) const {
        return o.data == data;
    }
    bool operator!=(const matrix& o) const {
        return o.data != data;
    }
    auto &operator[](int i){
        return data[i];
    }
    const auto& operator[](int i) const {
        return data[i];
    }


};


























/*

class vector{
    std::vector<number> data;
public:
    vector() = default;
    vector(int d): data(d){}
    explicit vector(std::vector<number> data): data(data){}

    */
/* vector addition *//*

    friend vector operator+(vector l, const vector& r);
    friend vector operator-(vector l, const vector& r);

    */
/* dot product*//*

    friend number operator*(const vector& l, const vector& r);

    */
/* multiply by scalar*//*

    friend vector operator*(vector v, const number& s);
    friend vector operator*(const number& s, vector v);
    friend vector operator*(vector v, int s);
    friend vector operator*(int s, vector v);

    */
/* multiply matrix *//*

    friend vector operator*(const vector& v, const matrix& m);
    friend vector operator*(const matrix& m, const vector& v);

    bool operator==(const vector& o) const {
        return o.data == data;
    }
    bool operator!=(const vector& o) const {
        return o.data != data;
    }
    number& operator[](int i){
        return data[i];
    }
    const number& operator[](int i) const {
        return data[i];
    }
    void push_back(const number& n) {
        data.push_back(n);
    }
    size_t size() const {
        return data.size();
    }
    friend class matrix;
};

class matrix{
    std::vector<vector> data;

public:
    matrix(int rows, int cols): data(rows, vector(cols)){}
    explicit matrix(std::vector<vector> data): data(data){}

    */
/* matrix addition *//*

    friend matrix operator+(matrix l, const matrix& r);

    */
/* multiply vector *//*

    friend vector operator*(const vector& v, const matrix& m);
    friend vector operator*(const matrix& m, const vector& v);

    vector col(int n) const;
    vector row(int n) const {
        return data[n];
    }

    */
/* eliminate matrix (self) to row echelon form, *
     * return determinant coeficient *//*

    number eliminate();

    */
/* reduce matrix (self) to reduced row echelon *
     * form from row echelon *//*

    void reduce();

    */
/* search for k-th pivot index, starting from h-th row
     * of matrix with m rows at total *//*

    int pivot_search(int h, int k, int m);

    int height() const {
        return data.size();
    }
    int width() const {
        return data[0].size();
    }
    bool operator==(const matrix& o) const {
        return o.data == data;
    }
    bool operator!=(const matrix& o) const {
        return o.data != data;
    }
    vector &operator[](int i){
        return data[i];
    }
    const vector& operator[](int i) const {
        return data[i];
    }

    void print() const{
        for (const auto& row : data){
            for (const auto& cell : row.data)
            {
                std::cout << to_string(cell) << '\t';
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

};
*/



void ordinal_argmax(const matrix& pred, std::vector<int>& ord_preds, int start);
double accuracy_score(const std::vector<int>& ord_pred, const std::vector<int>& y);




#endif //NN_D_H
